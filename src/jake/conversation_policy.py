"""Deterministic request routing and claim checks; no model-granted authority.

English claim patterns are defense in depth, not a proof about arbitrary prose.
No scene facts, private templates or tool credentials are available to generation.
"""

import re
import unicodedata
from enum import StrEnum

from jake.conversation_domain import ConversationContext

MAX_REPLY_CHARS = 800


class RestrictedClaim(StrEnum):
    OCCUPANCY = "occupancy"
    ACTION = "action"
    EMERGENCY = "emergency"
    IDENTITY = "identity"
    VISITOR = "visitor"
    SECRET = "secret"
    SENSOR = "sensor"
    FORMAT = "format"


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).replace("’", "'").casefold().split())


def matches(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def request_category(text: str) -> RestrictedClaim | None:
    value = normalized(text)
    if matches(
        value,
        (
            r"(?:^|[.!?]\s*)(?:is|are|was|were)\s+(?!(?:a|the|my|your|our|smart)\b)"
            r".{0,60}\b(?:home|inside|at home|in the house|here|present)\b",
            r"\b(?:who|anyone|anybody|someone)\b.{0,25}\b(?:home|inside|here|present)\b",
            r"\bwhere(?:'s| (?:is|are|was|were))\b",
            r"\b(?:who|how many people)\b.{0,25}\b(?:see|house|camera)\b",
            r"\b(?:tell|show|list|reveal)\b.{0,30}\b(?:occupancy|residents|visitors)\b",
        ),
    ):
        return RestrictedClaim.OCCUPANCY
    if matches(
        value,
        (
            r"\b(?:show|tell|reveal|print|repeat|give|read|what)\b.{0,70}"
            r"\b(?:system prompt|hidden prompt|password|secret|credential|encryption key|"
            r"biometric|embedding|resident templates?|visitor templates?)\b",
        ),
    ):
        return RestrictedClaim.SECRET
    command = r"(?:^|[.!?]\s*)(?:(?:hey )?jake[, ]+)?(?:please |(?:can|could|would|will) you )?"
    if value == "911" or matches(
        value,
        (
            command + r"(?:call|contact|notify|dispatch|send)\b.{0,35}\b(?:police|911|"
            r"ambulance|emergency|firefighters)\b",
        ),
    ):
        return RestrictedClaim.EMERGENCY
    if matches(
        value,
        (
            command
            + r"(?:unlock|lock|open|close|turn|switch|set|adjust|arm|disarm|activate)\b.{0,50}"
            r"\b(?:door|locks?|lights?|thermostat|temperature|alarm|heat|heating)\b",
            command + r"(?:search|browse|look up|google)\b",
        ),
    ):
        return RestrictedClaim.ACTION
    return None


def authorized_name(context: ConversationContext) -> str | None:
    speaker = context.speaker
    name = speaker.resident_name
    if (
        speaker.association_status == "visual_only"
        and speaker.identity_state == "RESIDENT"
        and speaker.resident_id
        and name
        and 1 <= len(name) <= 80
        and name.isprintable()
    ):
        return name
    return None


def rejected_claim(text: str, context: ConversationContext) -> RestrictedClaim | None:
    if not text.strip() or len(text) > MAX_REPLY_CHARS or not text.isprintable():
        return RestrictedClaim.FORMAT
    value = normalized(text)
    # Claims require subject/predicate relationships, not isolated topic words.
    rules: tuple[tuple[RestrictedClaim, tuple[str, ...]], ...] = (
        (
            RestrictedClaim.EMERGENCY,
            (
                r"\b(?:i|we|jake)(?:'ve|'ll)? (?:have |already |just |will |"
                r"can )*(?:called|contacted|notified|dispatched|call|contact|"
                r"notify|dispatch)\b.{0,40}\b(?:police|911|ambulance|emergency)\b",
                r"\b(?:police|911|ambulance|"
                r"emergency services)\b.{0,25}\b(?:called|contacted|notified|"
                r"dispatched|on the way)\b",
            ),
        ),
        (
            RestrictedClaim.ACTION,
            (
                r"\b(?:i|we|jake)(?:'ve|'ll)? (?:have |already |just |will |"
                r"can )*(?:unlocked|locked|opened|closed|turned|switched|set|"
                r"adjusted|armed|disarmed|activated|unlock|lock|open|close|turn|"
                r"switch|arm|disarm)\b.{0,35}\b"
                r"(?:door|locks?|lights?|thermostat|alarm|heating|temperature|it|them)\b",
                r"\b(?:door|lights?|thermostat|alarm)\b.{0,25}\b(?:has been|"
                r"have been|is now|are now|was|were)\b",
                r"\b(?:i|we)(?:'ve)? (?:have |just )*(?:searched|browsed|looked up|"
                r"checked online)\b",
                r"\b(?:i|we) (?:can|will) (?:control|browse|search the web)\b",
            ),
        ),
        (
            RestrictedClaim.OCCUPANCY,
            (
                r"\b(?:is|are|was|were|isn't|aren't|wasn't|weren't) (?:not |"
                r"currently |still |already )*(?:home|inside|at home|in the house|here)\b",
                r"\b(?:nobody|no one|everyone|someone|anyone) (?:is |was )?(?:home|inside|here)\b",
                r"\b(?:house|home) (?:is|was) (?:empty|occupied|unoccupied)\b",
                r"\bthere (?:is|are|was|were)\b.{0,30}\b"
                r"(?:people|person|someone|resident|visitor)\b",
                r"\b(?:lives?|stays?) (?:here|in this house|at this address)\b",
                r"\b\w+'s (?:home|inside|in the kitchen|in the bedroom)\b",
                r"\b(?:is|are) (?:in|at) the (?:kitchen|bedroom|living room|front door)\b",
                r"\b(?:i|we) (?:can |currently )*(?:see|detect|"
                r"observe)\b.{0,40}\b(?:person|people|resident|visitor|someone)\b",
            ),
        ),
        (
            RestrictedClaim.VISITOR,
            (
                r"\b(?:you|they|he|she|that person) (?:are|is|were|was|must be|"
                r"seem to be)\b.{0,20}\b(?:visitor|resident)\b",
                r"\b(?:your|their) (?:visitor|resident) (?:status|classification|id)\b",
                r"\b(?:first.time|frequent|recurring|known)_visitor\b",
            ),
        ),
        (
            RestrictedClaim.SECRET,
            (
                r"\b(?:system prompt|hidden prompt|encryption key|password|"
                r"credential|secret|embedding|resident templates?|visitor templates?)"
                r"\s*(?:is|are|:|=)",
                r"\b(?:here is|here are)\b.{0,30}\b(?:prompt|keys?|templates?|"
                r"embeddings?|credentials?)\b",
                r"-----begin .*private key",
            ),
        ),
        (
            RestrictedClaim.SENSOR,
            (
                r"\b(?:the|your|our)\b.{0,30}\bcamera (?:is|currently|records|"
                r"recorded|shows|detected|saw|has)\b",
                r"\b(?:package|parcel|delivery) (?:was|is|has been) (?:received|"
                r"delivered|detected)\b",
                r"\b(?:yesterday|earlier|last night)\b.{0,30}\b(?:you|resident|"
                r"visitor)\b.{0,20}\b(?:arrived|left|came|visited)\b",
            ),
        ),
    )
    for category, patterns in rules:
        if matches(value, patterns):
            return category
    capabilities = context.supported_actions
    if "resident recognition" not in capabilities and matches(
        value, (r"\bi (?:can|am able to) (?:recognize|identify) (?:enrolled )?residents\b",)
    ):
        return RestrictedClaim.ACTION
    if "visitor recognition" not in capabilities and matches(
        value,
        (
            r"\bi (?:can|am able to) (?:recognize|identify|track) (?:returning )?visitors\b",
            r"\b(?:i can|and) keep track of returning visitors\b",
        ),
    ):
        return RestrictedClaim.ACTION
    name = authorized_name(context)
    # Explicit identity assertions and direct forms of address require visual authority.
    if matches(
        value,
        (
            r"\byour name is\b",
            r"\bi (?:recognize|identified|remember) you\b",
        ),
    ):
        return RestrictedClaim.IDENTITY
    for match in re.finditer(
        r"\b(?:hello|hi|hey|good morning|good evening|good afternoon|nice to meet you)"
        r"[, ]+([\w'-]+)",
        value,
    ):
        addressed = match.group(1)
        if addressed not in {"there", "again", "how", "what", "what's", "it's"} and (
            name is None or addressed != normalized(name).split()[0]
        ):
            return RestrictedClaim.IDENTITY
    for match in re.finditer(r"\b[Yy]ou (?:are|must be|look like) ([A-Z][\w'-]+)", text):
        if name is None or normalized(match.group(1)) != normalized(name):
            return RestrictedClaim.IDENTITY
    # Vocatives ("Thanks, Joseph", "Joseph, ...") also need visual authority.
    addresses = re.findall(r",\s*([A-Z][\w'-]+)(?=[,.!?]|$)", text)
    addresses += re.findall(r"^([A-Z][\w'-]+),", text)
    for addressed in addresses:
        if addressed.casefold() not in {"yeah", "yes", "no", "okay", "well", "sure", "thanks"} and (
            name is None or normalized(addressed) != normalized(name)
        ):
            return RestrictedClaim.IDENTITY
    for match in re.finditer(r"\bI(?: am|'m) ([A-Z][\w'-]+)", text):
        if match.group(1) != "Jake":
            return RestrictedClaim.IDENTITY
    return None
