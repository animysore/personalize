"""User context data structures and formatting utilities."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class UserProfile:
    """Represents a user's profile and preferences."""

    user_id: str
    name: Optional[str] = None
    preferences: dict = field(default_factory=dict)
    attributes: dict = field(default_factory=dict)

    def to_text(self, style: str = "structured") -> str:
        """
        Convert profile to text representation.

        Args:
            style: Formatting style - "structured", "prose", or "minimal"

        Returns:
            Text representation of user profile
        """
        if style == "minimal":
            return self._to_minimal()
        elif style == "prose":
            return self._to_prose()
        return self._to_structured()

    def _to_structured(self) -> str:
        lines = []
        if self.name:
            lines.append(f"Name: {self.name}")
        if self.preferences:
            lines.append("Preferences:")
            for key, value in self.preferences.items():
                lines.append(f"  - {key}: {value}")
        if self.attributes:
            lines.append("Attributes:")
            for key, value in self.attributes.items():
                lines.append(f"  - {key}: {value}")
        return "\n".join(lines)

    def _to_prose(self) -> str:
        parts = []
        if self.name:
            parts.append(f"The user's name is {self.name}.")
        if self.preferences:
            prefs = ", ".join(f"{k}: {v}" for k, v in self.preferences.items())
            parts.append(f"Their preferences include {prefs}.")
        if self.attributes:
            attrs = ", ".join(f"{k} is {v}" for k, v in self.attributes.items())
            parts.append(f"About them: {attrs}.")
        return " ".join(parts)

    def _to_minimal(self) -> str:
        items = []
        if self.name:
            items.append(self.name)
        items.extend(f"{k}={v}" for k, v in self.preferences.items())
        items.extend(f"{k}={v}" for k, v in self.attributes.items())
        return "; ".join(items)


@dataclass
class HistoryItem:
    """A single item in user's interaction history."""

    content: str
    item_type: str = "interaction"  # interaction, preference, action, etc.
    timestamp: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class UserHistory:
    """User's interaction history."""

    user_id: str
    items: list[HistoryItem] = field(default_factory=list)

    def to_text(
        self,
        max_items: Optional[int] = None,
        style: str = "list",
        include_timestamps: bool = False,
    ) -> str:
        """
        Convert history to text representation.

        Args:
            max_items: Maximum number of items to include (most recent)
            style: Formatting style - "list", "prose", or "compact"
            include_timestamps: Whether to include timestamps

        Returns:
            Text representation of user history
        """
        items = self.items[-max_items:] if max_items else self.items

        if style == "compact":
            return self._to_compact(items)
        elif style == "prose":
            return self._to_prose(items, include_timestamps)
        return self._to_list(items, include_timestamps)

    def _to_list(self, items: list[HistoryItem], include_timestamps: bool) -> str:
        lines = []
        for item in items:
            line = f"- [{item.item_type}] {item.content}"
            if include_timestamps and item.timestamp:
                line = f"- [{item.timestamp}] [{item.item_type}] {item.content}"
            lines.append(line)
        return "\n".join(lines)

    def _to_prose(self, items: list[HistoryItem], include_timestamps: bool) -> str:
        parts = []
        for item in items:
            part = item.content
            if include_timestamps and item.timestamp:
                part = f"At {item.timestamp}, {item.content.lower()}"
            parts.append(part)
        return " ".join(parts)

    def _to_compact(self, items: list[HistoryItem]) -> str:
        return " | ".join(item.content for item in items)

    def add(
        self,
        content: str,
        item_type: str = "interaction",
        timestamp: Optional[str] = None,
        **metadata,
    ) -> None:
        """Add an item to history."""
        self.items.append(
            HistoryItem(
                content=content,
                item_type=item_type,
                timestamp=timestamp,
                metadata=metadata,
            )
        )


@dataclass
class UserContext:
    """Complete user context combining profile and history."""

    user_id: str
    profile: Optional[UserProfile] = None
    history: Optional[UserHistory] = None

    def to_text(
        self,
        include_profile: bool = True,
        include_history: bool = True,
        max_history_items: Optional[int] = None,
        profile_style: str = "structured",
        history_style: str = "list",
    ) -> str:
        """
        Convert full user context to text.

        Args:
            include_profile: Whether to include profile
            include_history: Whether to include history
            max_history_items: Max history items to include
            profile_style: Profile formatting style
            history_style: History formatting style

        Returns:
            Combined text representation
        """
        sections = []

        if include_profile and self.profile:
            profile_text = self.profile.to_text(style=profile_style)
            if profile_text:
                sections.append(f"## Profile\n{profile_text}")

        if include_history and self.history and self.history.items:
            history_text = self.history.to_text(
                max_items=max_history_items, style=history_style
            )
            if history_text:
                sections.append(f"## History\n{history_text}")

        return "\n\n".join(sections)

    @classmethod
    def from_dict(cls, data: dict) -> "UserContext":
        """Create UserContext from dictionary."""
        user_id = data.get("user_id", "unknown")

        profile = None
        if "profile" in data:
            profile = UserProfile(
                user_id=user_id,
                name=data["profile"].get("name"),
                preferences=data["profile"].get("preferences", {}),
                attributes=data["profile"].get("attributes", {}),
            )

        history = None
        if "history" in data:
            history = UserHistory(user_id=user_id)
            for item in data["history"]:
                if isinstance(item, str):
                    history.add(item)
                else:
                    history.add(
                        content=item.get("content", ""),
                        item_type=item.get("type", "interaction"),
                        timestamp=item.get("timestamp"),
                    )

        return cls(user_id=user_id, profile=profile, history=history)
