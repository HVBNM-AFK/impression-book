"""印象簿 — 亲密度与人际印象管理插件

让 Bot 全局识别每一位群成员，为每个人维护好感度与印象条目：
- 好感度由 Bot 根据近期发言自动微调，主人也可通过指令随时调整；
- 印象条目支持向量检索，在对话中按需注入聊天上下文，让 Bot "记得"对方；
- 主人通过 QQ 号认证，拥有全部管理指令。
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import asyncio
import json
import math
import os
import re

from json_repair import repair_json
from maibot_sdk import Command, EventHandler, Field, MaiBotPlugin, PluginConfigBase
from maibot_sdk.types import EventType

# ===== 常量 =====

_PROFILES_FILE = "profiles.json"
_IMPRESSIONS_FILE = "impressions.json"
_STORAGE_VERSION = 1

_LEVEL_NAMES = ("陌生", "相识", "熟悉", "朋友", "好友", "挚友")
_QQ_PATTERN = r"\d{5,12}"

# 自动评估的好感度单次最大变化幅度
_MAX_AUTO_DELTA = 2
# 每人消息缓冲区的最大长度（供自动评估取样）
_MESSAGE_BUFFER_LIMIT = 10


# ===== 配置模型 =====


class PluginSectionConfig(PluginConfigBase):
    """插件基础配置。"""

    __ui_label__ = "插件"
    __ui_icon__ = "package"
    __ui_order__ = 0

    enabled: bool = Field(default=False, description="是否启用插件")
    config_version: str = Field(default="1.0.0", description="配置版本")


class PersonaConfig(PluginConfigBase):
    """Bot 身份配置。"""

    __ui_label__ = "身份"
    __ui_icon__ = "user"
    __ui_order__ = 1

    bot_name: str = Field(default="麦麦", description="Bot 的名字，用于印象卡文案与检索触发词")


class OwnerConfig(PluginConfigBase):
    """主人配置。"""

    __ui_label__ = "主人"
    __ui_icon__ = "crown"
    __ui_order__ = 2

    owner_qq: str = Field(default="", description="主人 QQ 号，管理指令仅对其开放，留空则关闭管理功能")


class FavorabilityConfig(PluginConfigBase):
    """好感度与亲密等级配置。"""

    __ui_label__ = "好感度"
    __ui_icon__ = "heart"
    __ui_order__ = 3

    initial: int = Field(default=30, description="新成员的初始好感度（0-100）")
    threshold_acquaintance: int = Field(default=10, description="达到「相识」所需好感度")
    threshold_familiar: int = Field(default=30, description="达到「熟悉」所需好感度")
    threshold_friend: int = Field(default=50, description="达到「朋友」所需好感度")
    threshold_close: int = Field(default=70, description="达到「好友」所需好感度")
    threshold_bestie: int = Field(default=90, description="达到「挚友」所需好感度")


class RetrievalConfig(PluginConfigBase):
    """印象检索与注入配置。"""

    __ui_label__ = "检索注入"
    __ui_icon__ = "search"
    __ui_order__ = 4

    trigger_mode: str = Field(
        default="name_or_private",
        description="印象卡注入触发方式：name_or_private（提到 Bot 名字或私聊时）/ every_message（每条消息）/ off（关闭）",
    )
    top_k: int = Field(default=3, description="印象卡中最多展示的印象条数")


class AutoJudgeConfig(PluginConfigBase):
    """好感度自动评估配置。"""

    __ui_label__ = "自动评估"
    __ui_icon__ = "scale"
    __ui_order__ = 5

    enabled: bool = Field(default=True, description="是否由 Bot 根据发言自动微调好感度与印象")
    judge_every_n_messages: int = Field(default=8, description="每人累计多少条发言后触发一次评估")
    cooldown_minutes: int = Field(default=30, description="同一成员两次评估的最小间隔（分钟）")
    model_task: str = Field(default="", description="评估所用的模型任务名，留空使用宿主默认任务")
    max_auto_impressions: int = Field(default=20, description="每人自动生成的印象条目上限（主人设定的条目不受限）")


class ImpressionBookConfig(PluginConfigBase):
    """印象簿插件配置。"""

    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    persona: PersonaConfig = Field(default_factory=PersonaConfig)
    owner: OwnerConfig = Field(default_factory=OwnerConfig)
    favorability: FavorabilityConfig = Field(default_factory=FavorabilityConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    auto_judge: AutoJudgeConfig = Field(default_factory=AutoJudgeConfig)


# ===== 工具函数 =====


def _now_iso() -> str:
    """返回当前时间的 ISO 字符串。"""

    return datetime.now().isoformat(timespec="seconds")


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """计算两个向量的余弦相似度，纯 Python 实现以避免额外依赖。"""

    if len(vec_a) != len(vec_b) or not vec_a:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _clamp(value: int, low: int = 0, high: int = 100) -> int:
    """将整数限制在指定范围内。"""

    return max(low, min(high, value))


def _new_entry_id(entries: List[Dict[str, Any]]) -> int:
    """为印象条目生成递增编号。"""

    existing = [int(e.get("id", 0)) for e in entries]
    return (max(existing) if existing else 0) + 1


# ===== 存储层 =====


class ImpressionStore:
    """档案与印象的 JSON 文件存储。

    数据保存在插件私有数据目录（data/plugins/<plugin_id>）下：
    - profiles.json：成员档案（身份、好感度、专属标记、统计）
    - impressions.json：印象条目（文本、向量、来源、时间）
    所有写入先落临时文件再原子替换，避免中途断电产生半个 JSON。
    """

    def __init__(self, data_dir) -> None:
        self._data_dir = data_dir
        self._profiles_path = data_dir / _PROFILES_FILE
        self._impressions_path = data_dir / _IMPRESSIONS_FILE
        self._lock = asyncio.Lock()
        self.profiles: Dict[str, Dict[str, Any]] = {}
        self.impressions: Dict[str, List[Dict[str, Any]]] = {}

    def load(self) -> None:
        """从磁盘加载数据，文件不存在时初始化为空。"""

        self._data_dir.mkdir(parents=True, exist_ok=True)
        self.profiles = self._read_json(self._profiles_path, "profiles")
        self.impressions = self._read_json(self._impressions_path, "impressions")

    @staticmethod
    def _read_json(path, key: str) -> Any:
        if not path.is_file():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return {}
        data = payload.get(key)
        return data if isinstance(data, (dict, list)) else {}

    async def save_profiles(self) -> None:
        async with self._lock:
            self._write_json(self._profiles_path, "profiles", self.profiles)

    async def save_impressions(self) -> None:
        async with self._lock:
            self._write_json(self._impressions_path, "impressions", self.impressions)

    @staticmethod
    def _write_json(path, key: str, data: Any) -> None:
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump({"version": _STORAGE_VERSION, key: data}, f, ensure_ascii=False, indent=1)
        os.replace(tmp_path, path)


# ===== 主插件类 =====


class ImpressionBookPlugin(MaiBotPlugin):
    """印象簿插件主类。"""

    config_model = ImpressionBookConfig

    async def on_load(self) -> None:
        """加载数据并初始化运行状态。"""

        self._store = ImpressionStore(self.ctx.paths.data_dir)
        self._store.load()
        # 每人最近发言缓冲，供好感度自动评估取样
        self._message_buffers: Dict[str, List[str]] = {}
        # 每人距上次评估以来累计的消息数
        self._judge_counters: Dict[str, int] = {}
        # 每人上次评估时间戳（冷却控制）
        self._judge_cooldowns: Dict[str, float] = {}
        # 进行中的后台任务（评估任务 + 周期落盘任务）
        self._tasks: set = set()
        # 嵌入模型不可用标记：一旦确认不可用则停用向量检索，重载插件后重试
        self._embed_disabled = False
        # 档案有未落盘的变更时由周期任务统一保存，避免逐消息写盘
        self._profiles_dirty = False
        self._flush_task = asyncio.create_task(self._periodic_flush())
        self._track_task(self._flush_task)

    async def on_unload(self) -> None:
        """取消后台任务并落盘全部数据。"""

        for task in (self._flush_task, *self._tasks):
            task.cancel()
        self._tasks.clear()
        await self._store.save_profiles()
        await self._store.save_impressions()

    async def on_config_update(self, scope: str, config_data: dict[str, object], version: str) -> None:
        """处理配置热重载事件。"""

        del scope, config_data
        self.ctx.logger.info(f"印象簿配置已更新到版本 {version}")

    # ─── 内部工具 ────────────────────────────────────────────

    def _track_task(self, task: "asyncio.Task") -> None:
        """登记后台任务并在完成后自动移除。"""

        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _is_owner(self, user_id: str) -> bool:
        """通过 QQ 号校验主人身份。"""

        owner_qq = self.config.owner.owner_qq.strip()
        return bool(owner_qq) and user_id == owner_qq

    def _level_of(self, score: int) -> str:
        """按配置阈值将好感度映射为亲密等级名称。"""

        cfg = self.config.favorability
        thresholds = (
            cfg.threshold_acquaintance,
            cfg.threshold_familiar,
            cfg.threshold_friend,
            cfg.threshold_close,
            cfg.threshold_bestie,
        )
        level_index = 0
        for threshold in thresholds:
            if score >= threshold:
                level_index += 1
        return _LEVEL_NAMES[min(level_index, len(_LEVEL_NAMES) - 1)]

    def _get_or_create_profile(self, qq: str) -> Dict[str, Any]:
        """获取成员档案，不存在则按初始好感度建档。"""

        profile = self._store.profiles.get(qq)
        if profile is None:
            now = _now_iso()
            profile = {
                "qq": qq,
                "nickname": "",
                "cardname": "",
                "favorability": _clamp(self.config.favorability.initial),
                "is_special": False,
                "first_seen": now,
                "last_seen": now,
                "msg_count": 0,
                "group_ids": [],
            }
            self._store.profiles[qq] = profile
        return profile

    async def _embed_text(self, text: str) -> Optional[List[float]]:
        """生成文本向量；确认嵌入模型不可用后停用向量检索直到插件重载。"""

        if self._embed_disabled:
            return None
        try:
            result = await self.ctx.llm.embed(text=text)
        except Exception as exc:
            # 网络类瞬时错误不永久停用，仅本次跳过
            self.ctx.logger.warning(f"嵌入调用失败，本次跳过向量化: {exc}")
            return None
        if not isinstance(result, dict) or result.get("success") is False:
            self._embed_disabled = True
            error = result.get("error") if isinstance(result, dict) else "无返回"
            self.ctx.logger.warning(
                f"嵌入模型不可用（{error}），向量检索已停用；请在宿主模型配置中设置 embedding 任务后重载插件"
            )
            return None
        embedding = result.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            self._embed_disabled = True
            self.ctx.logger.warning("嵌入模型返回了空向量，向量检索已停用；请检查 embedding 任务配置后重载插件")
            return None
        return [float(x) for x in embedding]

    async def _periodic_flush(self) -> None:
        """周期性将内存中的档案变更落盘。"""

        while True:
            await asyncio.sleep(60)
            if self._profiles_dirty:
                self._profiles_dirty = False
                await self._store.save_profiles()

    # ─── 消息监听：建档 + 自动评估取样 + 印象卡注入 ────────────

    @EventHandler("impression_message_observer", description="识别群成员、累积评估样本、按需注入印象卡", event_type=EventType.ON_MESSAGE)
    async def handle_message(self, message: Any = None, stream_id: str = "", **kwargs: Any):
        """监听消息流，驱动建档、自动评估与印象卡注入。"""

        del kwargs

        if not isinstance(message, dict):
            return True, True, None, None, None

        user_info = message.get("user_info") or {}
        group_info = message.get("group_info") or {}
        qq = str(user_info.get("user_id") or "").strip()
        plain_text = str(message.get("plain_text") or "").strip()
        if not re.fullmatch(_QQ_PATTERN, qq):
            return True, True, None, None, None

        # 1. 建档/更新档案
        profile = self._get_or_create_profile(qq)
        profile["nickname"] = str(user_info.get("user_nickname") or "")
        profile["cardname"] = str(user_info.get("user_cardname") or "")
        profile["last_seen"] = _now_iso()
        profile["msg_count"] = int(profile.get("msg_count", 0)) + 1
        group_id = str(group_info.get("group_id") or "")
        if group_id and group_id not in profile["group_ids"]:
            profile["group_ids"].append(group_id)
        self._profiles_dirty = True

        # 指令消息不参与评估取样，避免指令文本污染印象
        if plain_text and not plain_text.startswith("/"):
            buffer = self._message_buffers.setdefault(qq, [])
            buffer.append(plain_text)
            if len(buffer) > _MESSAGE_BUFFER_LIMIT:
                del buffer[: len(buffer) - _MESSAGE_BUFFER_LIMIT]
            self._maybe_schedule_judge(qq)

        # 2. 按需注入印象卡
        if stream_id and self._should_inject(bool(group_info), plain_text):
            await self._inject_impression_card(qq, plain_text, stream_id)

        return True, True, None, None, None

    def _should_inject(self, is_group: bool, plain_text: str) -> bool:
        """判断当前消息是否触发印象卡注入。"""

        mode = self.config.retrieval.trigger_mode
        if mode == "off":
            return False
        if mode == "every_message":
            return True
        # name_or_private：私聊必触发；群聊中提到 Bot 名字时触发
        if not is_group:
            return True
        bot_name = self.config.persona.bot_name.strip()
        return bool(bot_name) and bot_name in plain_text

    async def _inject_impression_card(self, qq: str, plain_text: str, stream_id: str) -> None:
        """构建并注入印象卡到聊天上下文。"""

        profile = self._store.profiles.get(qq)
        if profile is None:
            return
        entries = self._store.impressions.get(qq, [])

        ranked = await self._rank_impressions(entries, plain_text)
        card = self._format_card(profile, ranked)
        try:
            await self.ctx.maisaka.append_context(
                stream_id,
                [{"type": "text", "content": card}],
                visible_text=card,
                source_kind="impression-book",
            )
        except Exception as exc:
            self.ctx.logger.warning(f"印象卡注入失败: {exc}")

    async def _rank_impressions(
        self, entries: List[Dict[str, Any]], plain_text: str
    ) -> List[Tuple[Dict[str, Any], Optional[float]]]:
        """印象条目排序：主人设定优先，其余按与当前消息的向量相关度排序。"""

        top_k = max(1, self.config.retrieval.top_k)
        owner_entries = [e for e in entries if e.get("source") == "owner"]
        other_entries = [e for e in entries if e.get("source") != "owner"]

        scored: List[Tuple[Dict[str, Any], Optional[float]]] = []
        query_vector = await self._embed_text(plain_text) if plain_text else None
        if query_vector is not None:
            for entry in other_entries:
                vector = entry.get("vector")
                score = _cosine_similarity(query_vector, vector) if isinstance(vector, list) else None
                scored.append((entry, score))
            # 有相似度的排前面；无向量的条目按时间倒序兜底
            scored.sort(key=lambda item: (item[1] is not None, item[1] or 0.0), reverse=True)
        else:
            other_entries = sorted(other_entries, key=lambda e: str(e.get("created_at") or ""), reverse=True)
            scored = [(entry, None) for entry in other_entries]

        ranked = [(entry, None) for entry in owner_entries] + scored
        return ranked[:top_k]

    def _format_card(self, profile: Dict[str, Any], ranked: List[Tuple[Dict[str, Any], Optional[float]]]) -> str:
        """生成注入上下文的印象卡文本。"""

        bot_name = self.config.persona.bot_name.strip() or "Bot"
        nickname = profile.get("nickname") or profile.get("qq") or "对方"
        score = int(profile.get("favorability", 0))
        level = self._level_of(score)
        special_tag = "，专属用户" if profile.get("is_special") else ""
        lines = [f"【{bot_name}对 {nickname} 的印象】好感度 {score}/100，关系：{level}{special_tag}"]
        for index, (entry, score_hint) in enumerate(ranked, start=1):
            similarity = f"（相关度 {score_hint:.2f}）" if score_hint is not None else ""
            lines.append(f"{index}. {entry.get('text', '')}{similarity}")
        return "\n".join(lines)

    # ─── 好感度自动评估 ──────────────────────────────────────

    def _maybe_schedule_judge(self, qq: str) -> None:
        """累计到阈值且冷却结束时，为该成员安排一次好感度评估。"""

        cfg = self.config.auto_judge
        if not cfg.enabled:
            return
        count = self._judge_counters.get(qq, 0) + 1
        self._judge_counters[qq] = count
        if count < max(1, cfg.judge_every_n_messages):
            return
        self._judge_counters[qq] = 0

        now = asyncio.get_running_loop().time()
        last = self._judge_cooldowns.get(qq, 0.0)
        if now - last < max(0, cfg.cooldown_minutes) * 60:
            return
        self._judge_cooldowns[qq] = now

        self._track_task(asyncio.create_task(self._run_judge(qq)))

    async def _run_judge(self, qq: str) -> None:
        """调用 LLM 评估某成员近期发言，微调好感度并提炼新印象。"""

        bot_name = self.config.persona.bot_name.strip() or "Bot"
        profile = self._store.profiles.get(qq)
        lines = self._message_buffers.get(qq, [])[-_MESSAGE_BUFFER_LIMIT:]
        if profile is None or not lines:
            return

        nickname = profile.get("nickname") or qq
        chat_lines = "\n".join(f"- {line}" for line in lines)
        prompt = (
            f"你是 QQ 群聊机器人「{bot_name}」的人际关系评估员。\n"
            f"根据群成员「{nickname}」的最近发言，评估 Ta 对 {bot_name} 及群内氛围的态度变化，"
            f"并判断是否有值得长期记住的新印象（如爱好、习惯、专长、约定）。\n\n"
            f"最近发言：\n{chat_lines}\n\n"
            "严格输出 JSON，不要输出任何其他内容：\n"
            '{"delta": -2到2的整数，表示好感度变化，无变化为0, '
            '"reason": "一句话原因", '
            '"impression_note": "新印象的一句话描述，没有则为空字符串"}'
        )

        task_name = self.config.auto_judge.model_task.strip()
        try:
            if task_name:
                result = await self.ctx.llm.generate(prompt, task_name=task_name, temperature=0.3, max_tokens=300)
            else:
                result = await self.ctx.llm.generate(prompt, temperature=0.3, max_tokens=300)
        except Exception as exc:
            self.ctx.logger.warning(f"好感度评估调用失败: {exc}")
            return

        response_text = ""
        if isinstance(result, dict):
            response_text = str(result.get("response") or "")
        try:
            judgment = json.loads(str(repair_json(response_text)))
        except Exception:
            self.ctx.logger.warning(f"好感度评估结果无法解析: {response_text[:120]}")
            return
        if not isinstance(judgment, dict):
            return

        # 好感度微调：限制单次幅度并收敛到 0-100
        try:
            delta = int(judgment.get("delta") or 0)
        except (TypeError, ValueError):
            delta = 0
        delta = max(-_MAX_AUTO_DELTA, min(_MAX_AUTO_DELTA, delta))
        reason = str(judgment.get("reason") or "").strip()
        if delta != 0:
            old_score = int(profile.get("favorability", 0))
            new_score = _clamp(old_score + delta)
            profile["favorability"] = new_score
            self._profiles_dirty = True
            self.ctx.logger.info(
                f"好感度自动调整：{nickname}({qq}) {old_score} -> {new_score}（{reason or '无说明'}）"
            )

        # 提炼新印象：自动条目受上限约束，主人条目不受影响
        note = str(judgment.get("impression_note") or "").strip()
        if note:
            await self._append_impression(qq, note, source="auto")

    async def _append_impression(self, qq: str, text: str, source: str) -> Dict[str, Any]:
        """新增印象条目并尽力向量化，返回新条目。"""

        entries = self._store.impressions.setdefault(qq, [])
        vector = await self._embed_text(text)
        entry = {
            "id": _new_entry_id(entries),
            "text": text,
            "vector": vector,
            "source": source,
            "created_at": _now_iso(),
        }
        entries.append(entry)

        # 自动印象超限时移除最旧的自动条目
        max_auto = max(1, self.config.auto_judge.max_auto_impressions)
        auto_entries = [e for e in entries if e.get("source") == "auto"]
        overflow = len(auto_entries) - max_auto
        if overflow > 0:
            auto_entries.sort(key=lambda e: str(e.get("created_at") or ""))
            drop_ids = {e["id"] for e in auto_entries[:overflow]}
            self._store.impressions[qq] = [e for e in entries if e.get("id") not in drop_ids]

        await self._store.save_impressions()
        return entry

    # ─── 指令：帮助与查询 ────────────────────────────────────

    @Command("impression_help", description="印象簿指令帮助", pattern=r"^/印象帮助$")
    async def handle_help(self, stream_id: str = "", **kwargs: Any):
        """输出指令帮助；主人可见完整管理指令。"""

        user_id = str(kwargs.get("user_id") or "")
        if not self._is_owner(user_id):
            await self.ctx.send.text("【印象簿】\n/印象 — 查看自己的印象卡", stream_id)
            return True, "已发送帮助", True

        bot_name = self.config.persona.bot_name.strip() or "Bot"
        help_text = (
            f"【印象簿指令帮助】（{bot_name} 的主人专属）\n"
            "/印象 [QQ号] — 查看印象卡，不带 QQ 号查自己\n"
            "/加印象 <QQ号> <内容> — 追加印象条目\n"
            "/改印象 <QQ号> <编号> <内容> — 修改指定条目\n"
            "/删印象 <QQ号> <编号> — 删除指定条目\n"
            "/好感 <QQ号> <+N|-N|N> — 增减或直接设定好感度\n"
            "/专属 <QQ号> 开|关 — 标记或取消专属用户\n"
            "/亲密榜 — 好感度前十排行\n"
            "/忘记 <QQ号> — 删除该成员全部数据\n"
            "/印象帮助 — 本帮助"
        )
        await self.ctx.send.text(help_text, stream_id)
        return True, "已发送帮助", True

    @Command("impression_card", description="查看印象卡", pattern=rf"^/印象(?:\s+(?P<qq>{_QQ_PATTERN}))?$")
    async def handle_card(self, stream_id: str = "", **kwargs: Any):
        """查看自己或指定成员的印象卡（查他人仅主人）。"""

        user_id = str(kwargs.get("user_id") or "")
        matched = kwargs.get("matched_groups") if isinstance(kwargs.get("matched_groups"), dict) else {}
        target_qq = str(matched.get("qq") or "").strip()

        if target_qq and not self._is_owner(user_id):
            await self.ctx.send.text("只有主人可以查看他人的印象卡", stream_id)
            return True, "权限不足", True
        if not target_qq:
            target_qq = user_id

        profile = self._store.profiles.get(target_qq)
        if profile is None:
            await self.ctx.send.text(f"印象簿中还没有 {target_qq} 的档案，聊几句就有了", stream_id)
            return True, "档案不存在", True

        entries = self._store.impressions.get(target_qq, [])
        ranked = [(entry, None) for entry in entries]
        card = self._format_card(profile, ranked[: max(1, self.config.retrieval.top_k)])
        await self.ctx.send.text(card, stream_id)
        return True, "已发送印象卡", True

    @Command("impression_top", description="好感度排行榜", pattern=r"^/亲密榜$")
    async def handle_top(self, stream_id: str = "", **kwargs: Any):
        """好感度前十排行（仅主人）。"""

        if not self._is_owner(str(kwargs.get("user_id") or "")):
            await self.ctx.send.text("只有主人可以查看亲密榜", stream_id)
            return True, "权限不足", True

        if not self._store.profiles:
            await self.ctx.send.text("印象簿还是空的", stream_id)
            return True, "暂无数据", True

        ranked = sorted(
            self._store.profiles.values(),
            key=lambda p: int(p.get("favorability", 0)),
            reverse=True,
        )[:10]
        lines = ["【亲密榜】"]
        for index, profile in enumerate(ranked, start=1):
            nickname = profile.get("nickname") or profile.get("qq") or "未知"
            score = int(profile.get("favorability", 0))
            special = " [专属]" if profile.get("is_special") else ""
            lines.append(f"{index}. {nickname}({profile.get('qq')}) — {score}/100 · {self._level_of(score)}{special}")
        await self.ctx.send.text("\n".join(lines), stream_id)
        return True, "已发送亲密榜", True

    # ─── 指令：印象管理（仅主人） ─────────────────────────────

    @Command("impression_add", description="追加印象条目", pattern=rf"^/加印象\s+(?P<qq>{_QQ_PATTERN})\s+(?P<content>.+)$")
    async def handle_add(self, stream_id: str = "", **kwargs: Any):
        """追加一条印象并尝试向量化（仅主人）。"""

        if not self._is_owner(str(kwargs.get("user_id") or "")):
            await self.ctx.send.text("只有主人可以修改印象", stream_id)
            return True, "权限不足", True

        matched = kwargs.get("matched_groups") if isinstance(kwargs.get("matched_groups"), dict) else {}
        qq = str(matched.get("qq") or "").strip()
        content = str(matched.get("content") or "").strip()
        if not content:
            await self.ctx.send.text("印象内容不能为空", stream_id)
            return True, "内容为空", True

        profile = self._get_or_create_profile(qq)
        self._profiles_dirty = True
        entry = await self._append_impression(qq, content, source="owner")
        nickname = profile.get("nickname") or qq
        await self.ctx.send.text(f"已为 {nickname}({qq}) 记下第 {entry['id']} 条印象：{content}", stream_id)
        return True, "印象已添加", True

    @Command(
        "impression_edit",
        description="修改印象条目",
        pattern=rf"^/改印象\s+(?P<qq>{_QQ_PATTERN})\s+(?P<entry_id>\d+)\s+(?P<content>.+)$",
    )
    async def handle_edit(self, stream_id: str = "", **kwargs: Any):
        """修改指定编号的印象条目（仅主人），修改后视为主人设定。"""

        if not self._is_owner(str(kwargs.get("user_id") or "")):
            await self.ctx.send.text("只有主人可以修改印象", stream_id)
            return True, "权限不足", True

        matched = kwargs.get("matched_groups") if isinstance(kwargs.get("matched_groups"), dict) else {}
        qq = str(matched.get("qq") or "").strip()
        entry_id = int(matched.get("entry_id") or 0)
        content = str(matched.get("content") or "").strip()

        entries = self._store.impressions.get(qq, [])
        entry = next((e for e in entries if int(e.get("id", 0)) == entry_id), None)
        if entry is None:
            await self.ctx.send.text(f"{qq} 没有编号为 {entry_id} 的印象条目", stream_id)
            return True, "条目不存在", True

        entry["text"] = content
        entry["source"] = "owner"
        entry["vector"] = await self._embed_text(content)
        await self._store.save_impressions()
        await self.ctx.send.text(f"已更新 {qq} 的第 {entry_id} 条印象：{content}", stream_id)
        return True, "印象已更新", True

    @Command(
        "impression_delete",
        description="删除印象条目",
        pattern=rf"^/删印象\s+(?P<qq>{_QQ_PATTERN})\s+(?P<entry_id>\d+)$",
    )
    async def handle_delete(self, stream_id: str = "", **kwargs: Any):
        """删除指定编号的印象条目（仅主人）。"""

        if not self._is_owner(str(kwargs.get("user_id") or "")):
            await self.ctx.send.text("只有主人可以删除印象", stream_id)
            return True, "权限不足", True

        matched = kwargs.get("matched_groups") if isinstance(kwargs.get("matched_groups"), dict) else {}
        qq = str(matched.get("qq") or "").strip()
        entry_id = int(matched.get("entry_id") or 0)

        entries = self._store.impressions.get(qq, [])
        remaining = [e for e in entries if int(e.get("id", 0)) != entry_id]
        if len(remaining) == len(entries):
            await self.ctx.send.text(f"{qq} 没有编号为 {entry_id} 的印象条目", stream_id)
            return True, "条目不存在", True

        self._store.impressions[qq] = remaining
        await self._store.save_impressions()
        await self.ctx.send.text(f"已删除 {qq} 的第 {entry_id} 条印象", stream_id)
        return True, "印象已删除", True

    # ─── 指令：好感度与专属标记（仅主人） ──────────────────────

    @Command(
        "favorability_adjust",
        description="调整或设定好感度",
        pattern=rf"^/好感\s+(?P<qq>{_QQ_PATTERN})\s+(?P<value>[+-]?\d{{1,3}})$",
    )
    async def handle_favorability(self, stream_id: str = "", **kwargs: Any):
        """增减（+N/-N）或直接设定（N）好感度（仅主人）。"""

        if not self._is_owner(str(kwargs.get("user_id") or "")):
            await self.ctx.send.text("只有主人可以调整好感度", stream_id)
            return True, "权限不足", True

        matched = kwargs.get("matched_groups") if isinstance(kwargs.get("matched_groups"), dict) else {}
        qq = str(matched.get("qq") or "").strip()
        raw_value = str(matched.get("value") or "0")

        profile = self._get_or_create_profile(qq)
        old_score = int(profile.get("favorability", 0))
        if raw_value.startswith(("+", "-")):
            new_score = _clamp(old_score + int(raw_value))
        else:
            new_score = _clamp(int(raw_value))
        profile["favorability"] = new_score
        await self._store.save_profiles()

        nickname = profile.get("nickname") or qq
        bot_name = self.config.persona.bot_name.strip() or "Bot"
        await self.ctx.send.text(
            f"{bot_name} 对 {nickname}({qq}) 的好感度：{old_score} -> {new_score}（{self._level_of(new_score)}）",
            stream_id,
        )
        return True, "好感度已调整", True

    @Command(
        "special_toggle",
        description="标记或取消专属用户",
        pattern=rf"^/专属\s+(?P<qq>{_QQ_PATTERN})\s+(?P<switch>开|关)$",
    )
    async def handle_special(self, stream_id: str = "", **kwargs: Any):
        """标记或取消专属用户（仅主人）。"""

        if not self._is_owner(str(kwargs.get("user_id") or "")):
            await self.ctx.send.text("只有主人可以设置专属用户", stream_id)
            return True, "权限不足", True

        matched = kwargs.get("matched_groups") if isinstance(kwargs.get("matched_groups"), dict) else {}
        qq = str(matched.get("qq") or "").strip()
        enabled = matched.get("switch") == "开"

        profile = self._get_or_create_profile(qq)
        profile["is_special"] = enabled
        await self._store.save_profiles()

        nickname = profile.get("nickname") or qq
        state = "已标记为专属用户" if enabled else "已取消专属标记"
        await self.ctx.send.text(f"{nickname}({qq}) {state}", stream_id)
        return True, "专属状态已更新", True

    @Command("profile_forget", description="删除成员全部数据", pattern=rf"^/忘记\s+(?P<qq>{_QQ_PATTERN})$")
    async def handle_forget(self, stream_id: str = "", **kwargs: Any):
        """删除指定成员的档案与全部印象（仅主人）。"""

        if not self._is_owner(str(kwargs.get("user_id") or "")):
            await self.ctx.send.text("只有主人可以执行忘记操作", stream_id)
            return True, "权限不足", True

        matched = kwargs.get("matched_groups") if isinstance(kwargs.get("matched_groups"), dict) else {}
        qq = str(matched.get("qq") or "").strip()

        existed = qq in self._store.profiles or qq in self._store.impressions
        self._store.profiles.pop(qq, None)
        self._store.impressions.pop(qq, None)
        self._message_buffers.pop(qq, None)
        self._judge_counters.pop(qq, None)
        self._judge_cooldowns.pop(qq, None)
        if existed:
            await self._store.save_profiles()
            await self._store.save_impressions()
            await self.ctx.send.text(f"已忘记 {qq} 的全部档案与印象", stream_id)
            return True, "已删除", True
        await self.ctx.send.text(f"印象簿中没有 {qq} 的数据", stream_id)
        return True, "无数据", True


def create_plugin() -> ImpressionBookPlugin:
    """创建插件实例（宿主程序入口）。"""

    return ImpressionBookPlugin()
