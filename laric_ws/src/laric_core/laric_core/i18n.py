"""
LARIC System - Internationalization (i18n)
Copyright (c) 2026 LARIC. All rights reserved.

Single source of truth for translation. Exposes 'set_language' and a
translator '_' that resolves the active gettext catalog on every call, so a
runtime language switch is observed by every module that imported '_'.
"""

import gettext
import os


# ==============================================================================
# PATHS
# ==============================================================================

# Resolve <project_root>/locales relative to this file's location:
#   - First "..": exit the package folder (where i18n.py lives)
#   - Second "..": exit the ROS 2 package folder
#   - Third "..": exit the 'src' directory
_BASE_DIR: str    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT: str = os.path.abspath(os.path.join(_BASE_DIR, "..", "..", ".."))
LOCALE_DIR: str   = os.path.join(_PROJECT_ROOT, "locales")

# gettext catalog domain — must match the .mo filename (laric.mo).
DOMAIN: str = "laric"

# Default language at process start. Overridden at runtime via set_language().
DEFAULT_LANGUAGE: str = "en"


# ==============================================================================
# RUNTIME STATE
# ==============================================================================

# Module-level so set_language()'s reassignment is seen by every '_' caller.
_current_language: str = DEFAULT_LANGUAGE
_translator: gettext.NullTranslations = gettext.NullTranslations()


# ==============================================================================
# PUBLIC API
# ==============================================================================

def set_language(lang: str) -> None:
    """
    Activates the gettext catalog for the given language code.

    Args:
        lang: ISO 639-1 language code matching a subdirectory in 'LOCALE_DIR'
            (e.g., "en", "es"). Falls back silently to the untranslated
            source strings if no catalog is found.
    """
    global _current_language, _translator
    _current_language = lang
    _translator = gettext.translation(
        domain=DOMAIN,
        localedir=LOCALE_DIR,
        languages=[lang],
        fallback=True,
    )


def get_language() -> str:
    """
    Returns the currently active language code (e.g., "en", "es").
    """
    return _current_language


def _(message: str) -> str:
    """
    Translates 'message' using the currently active catalog.

    Args:
        message: Source-language (English) message identifier exactly as
            it appears in the .po files.

    Returns:
        The translated string, or 'message' itself if no translation is
        available.
    """
    return _translator.gettext(message)


# Initialise with the default language so '_' is usable immediately on import.
set_language(DEFAULT_LANGUAGE)