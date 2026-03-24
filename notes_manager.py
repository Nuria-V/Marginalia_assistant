"""
notes_manager.py
================
Core module for capturing, validating, saving, and loading
personal reading notes for the Literary Analysis Pipeline.

Used across all pipeline phases:
    from notes_manager import NotesManager
"""

import logging
import os
import uuid
from datetime import date

import pandas as pd


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
# We create a module-level logger named after this file ("notes_manager").
# Each phase that imports this module will see messages tagged with this name,
# making it easy to trace which module generated each log entry.
#
# Example output:
#   INFO:notes_manager:NotesManager ready. Notes loaded: 10
#   WARNING:notes_manager:Column 'tags' missing — adding empty column.
#   ERROR:notes_manager:File not found: /content/drive/MyDrive/user_notes.csv


logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

NOTES_FILENAME = "user_notes.csv"

# Minimum note length to provide meaningful NLP signal
MIN_NOTE_LENGTH = 20

MIN_RATING = 1
MAX_RATING = 5

# Column schema — order matters for readability and downstream compatibility
COLUMNS = [
    "note_id",      # Auto-generated short UUID
    "title",        # Book title
    "author",       # Author name
    "note_text",    # Personal reflection (primary NLP input)
    "rating",       # Integer score 1–5
    "tags",         # Comma-separated personal tags
    "date_read",    # Format: YYYY-MM-DD
    "date_added",   # Auto-set on creation
    "processed",    # Pipeline flag: False = pending vectorization
]


# =============================================================================
# MAIN CLASS
# =============================================================================

class NotesManager:
    """
    Manages the full lifecycle of user reading notes:
    create -> validate -> save -> load -> export to pipeline.

    Usage:
        nm = NotesManager(drive_path="/content/drive/MyDrive/project/")
        nm.add_note("Dune", "Frank Herbert", "Epic and philosophical...", 5, "scifi, epic")
        df = nm.get_notes_dataframe()
    """

    def __init__(self, drive_path: str = "/content/drive/MyDrive/"):
        """
        Args:
            drive_path: Path to the project folder in Google Drive.
                        Must match the folder containing books_clean.csv.
                        Example: "/content/drive/MyDrive/Colab Notebooks/Project/"
        """
        self.drive_path = drive_path
        self.notes_path = os.path.join(drive_path, NOTES_FILENAME)
        self.notes_df = self._load_or_create()

        logger.info("NotesManager ready. Notes file: %s | Notes loaded: %d",
                    self.notes_path, len(self.notes_df))


    # -------------------------------------------------------------------------
    # PRIVATE METHODS
    # -------------------------------------------------------------------------

    def _load_or_create(self) -> pd.DataFrame:
        """
        Loads the notes CSV if it exists; otherwise creates an empty one.

        Returns:
            DataFrame with existing notes, or empty DataFrame on first run.
        """
        if os.path.exists(self.notes_path):
            df = pd.read_csv(self.notes_path, dtype=str)

            # Forward-compatible: add missing columns if schema has evolved
            for col in COLUMNS:
                if col not in df.columns:
                    logger.warning("Column '%s' missing in CSV — adding empty column.", col)
                    df[col] = ""

            logger.info("Notes CSV loaded from: %s", self.notes_path)
            return df

        else:
            df = pd.DataFrame(columns=COLUMNS)
            df.to_csv(self.notes_path, index=False)
            logger.info("New notes file created at: %s", self.notes_path)
            return df


    def _generate_id(self) -> str:
        """Returns a unique 8-character hex ID for each note."""
        return str(uuid.uuid4()).replace("-", "")[:8]


    def _validate_note(
        self, title: str, author: str, note_text: str, rating: int, date_read: str
    ) -> tuple:
        """
        Validates all required fields before saving.

        Returns:
            (True, "") if valid, or (False, error_message) if not.
        """
        if not title or not title.strip():
            return False, "Title cannot be empty."

        if not author or not author.strip():
            return False, "Author cannot be empty."

        if not note_text or len(note_text.strip()) < MIN_NOTE_LENGTH:
            char_count = len(note_text.strip()) if note_text else 0
            return False, (
                f"Note must be at least {MIN_NOTE_LENGTH} characters "
                f"(current: {char_count}). Write at least 1-2 sentences."
            )

        if not isinstance(rating, int) or not (MIN_RATING <= rating <= MAX_RATING):
            return False, f"Rating must be an integer between {MIN_RATING} and {MAX_RATING}."

        if date_read:
            try:
                pd.to_datetime(date_read, format="%Y-%m-%d")
            except ValueError:
                return False, "Date must be in YYYY-MM-DD format. Example: 2024-03-15"

        return True, ""


    def _save(self):
        """Persists the current DataFrame to Drive. Called after every mutation."""
        self.notes_df.to_csv(self.notes_path, index=False)
        logger.debug("Notes saved to: %s", self.notes_path)


    # -------------------------------------------------------------------------
    # PUBLIC METHODS
    # -------------------------------------------------------------------------

    def add_note(
        self,
        title: str,
        author: str,
        note_text: str,
        rating: int,
        tags: str = "",
        date_read: str = ""
    ) -> bool:
        """
        Adds a new note and saves it to Drive.

        Args:
            title:     Book title. Example: "Dune"
            author:    Author name. Example: "Frank Herbert"
            note_text: Personal reflection. Minimum 20 characters.
            rating:    Score 1-5 (integers only).
            tags:      Comma-separated tags (optional). Example: "scifi, epic, ecology"
            date_read: Read date in YYYY-MM-DD format (optional, defaults to today).

        Returns:
            True if saved successfully, False if validation failed.
        """
        if not date_read:
            date_read = str(date.today())

        is_valid, error_msg = self._validate_note(title, author, note_text, rating, date_read)
        if not is_valid:
            logger.error("Validation failed for '%s': %s", title, error_msg)
            return False

        new_note = {
            "note_id":    self._generate_id(),
            "title":      title.strip(),
            "author":     author.strip(),
            "note_text":  note_text.strip(),
            "rating":     rating,
            "tags":       tags.strip().lower(),
            "date_read":  date_read,
            "date_added": str(date.today()),
            "processed":  False,
        }

        self.notes_df = pd.concat(
            [self.notes_df, pd.DataFrame([new_note])],
            ignore_index=True
        )
        self._save()

        logger.info("Note saved: '%s' by %s (ID: %s)", title, author, new_note["note_id"])
        return True


    def show_notes(self, limit: int = None):
        """
        Prints notes.
        Intended for interactive use in Colab notebooks.

        Args:
            limit: Max number of notes to display. Displays all if None.
        """
        if len(self.notes_df) == 0:
            print("No notes saved yet. Use add_note() to add your first one.")
            return

        df_to_show = self.notes_df if limit is None else self.notes_df.head(limit)

        print(f"\n{'='*60}")
        print(f"YOUR READING NOTES  ({len(self.notes_df)} total)")
        print(f"{'='*60}\n")

        for idx, row in df_to_show.iterrows():
            stars = "*" * int(row["rating"]) if str(row["rating"]).isdigit() else "-"
            print(f"[{idx+1}] {row['title']}")
            print(f"    Author : {row['author']}")
            print(f"    Rating : {stars} ({row['rating']}/5)")
            print(f"    Note   : {row['note_text']}")
            if row["tags"] and str(row["tags"]) != "nan":
                print(f"    Tags   : {row['tags']}")
            if row["date_read"] and str(row["date_read"]) != "nan":
                print(f"    Read   : {row['date_read']}")
            print(f"    ID     : {row['note_id']}")
            print()


    def delete_note(self, note_id: str) -> bool:
        """
        Deletes a note by its ID.

        Args:
            note_id: The note ID shown by show_notes(). Example: "a3f9c2b1"

        Returns:
            True if deleted, False if ID not found.
        """
        mask = self.notes_df["note_id"] == note_id

        if mask.sum() == 0:
            logger.warning("delete_note: no note found with ID '%s'.", note_id)
            return False

        title = self.notes_df.loc[mask, "title"].values[0]
        self.notes_df = self.notes_df[~mask].reset_index(drop=True)
        self._save()

        logger.info("Note deleted: '%s' (ID: %s)", title, note_id)
        return True


    def get_notes_dataframe(self, only_unprocessed: bool = False) -> pd.DataFrame:
        """
        Exports notes as a DataFrame for downstream pipeline phases
        (NLP, Clustering, RAG).

        Args:
            only_unprocessed: If True, returns only notes not yet vectorized.

        Returns:
            DataFrame with the requested notes.

        Usage in Phase 3:
            df = nm.get_notes_dataframe()
            df_new = nm.get_notes_dataframe(only_unprocessed=True)
        """
        if len(self.notes_df) == 0:
            logger.warning("get_notes_dataframe called but no notes exist.")
            return pd.DataFrame(columns=COLUMNS)

        if only_unprocessed:
            mask = self.notes_df["processed"].astype(str).isin(["False", "false", "0", ""])
            result = self.notes_df[mask].copy()
            logger.info("Exporting %d unprocessed notes (of %d total).",
                        len(result), len(self.notes_df))
            return result

        logger.info("Exporting all %d notes.", len(self.notes_df))
        return self.notes_df.copy()


    def mark_as_processed(self, note_ids: list):
        """
        Marks notes as processed by the NLP pipeline.

        Args:
            note_ids: List of IDs to mark. Example: ["a3f9c2b1", "b7e2d9f0"]
        """
        mask = self.notes_df["note_id"].isin(note_ids)
        self.notes_df.loc[mask, "processed"] = True
        self._save()
        logger.info("%d notes marked as processed.", mask.sum())


    def get_stats(self) -> dict:
        """
        Returns a statistics summary of the notes collection.
        Also prints the summary for interactive use.

        Returns:
            dict with keys: total, avg_rating, avg_note_length, top_tags
        """
        if len(self.notes_df) == 0:
            logger.warning("get_stats called but no notes exist.")
            return {}

        ratings = pd.to_numeric(self.notes_df["rating"], errors="coerce")

        all_tags = []
        for tags_cell in self.notes_df["tags"].dropna():
            all_tags.extend([t.strip() for t in str(tags_cell).split(",") if t.strip()])

        top_tags = (
            pd.Series(all_tags).value_counts().head(5).to_dict() if all_tags else {}
        )

        stats = {
            "total":           len(self.notes_df),
            "avg_rating":      round(ratings.mean(), 1),
            "avg_note_length": int(self.notes_df["note_text"].str.len().mean()),
            "top_tags":        top_tags,
        }

        print(f"\n{'='*50}")
        print("NOTES STATISTICS")
        print(f"{'='*50}")
        print(f"Total notes     : {stats['total']}")
        print(f"Average rating  : {stats['avg_rating']} / 5.0")
        print(f"Avg note length : {stats['avg_note_length']} characters")
        if top_tags:
            print("Top 5 tags      :")
            for tag, count in top_tags.items():
                print(f"  {tag}: {count}")
        print(f"{'='*50}\n")

        logger.debug("get_stats called. Total notes: %d", stats["total"])
        return stats
