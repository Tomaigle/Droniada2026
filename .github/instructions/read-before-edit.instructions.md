---
description: "Use when reviewing, editing, or suggesting improvements to any file in Droniada2026. Always read the file content first, and reference Regulamin_konkursu_Droniada_Challenge_2026.pdf for project constraints."
applyTo: "**"
---

# Always Read Files Before Editing or Suggesting Changes

## Core Rule

Before making ANY code changes or suggesting refactors:
1. **Read the entire file** using `read_file` to see the current state
2. **Understand context** — imports, dependencies, class structure, function signatures
3. **Identify patterns** — existing coding style, conventions, variable naming in that specific file
4. **Then proceed** — edit or suggest based on what you actually see

## Why This Matters

- Avoids suggesting changes that conflict with existing code structure
- Ensures edits are precise and maintain file consistency
- Prevents redundant changes or breaking existing logic
- Catches edge cases and side effects in the actual code

## What "Read Before Edit" Covers

✅ **Do apply:**
- Before `replace_string_in_file` or `multi_replace_string_in_file`
- Before suggesting refactors, optimizations, or style improvements
- Before importing code or cross-referencing between files
- When the file context has changed from previous operations

❌ **Don't need to:**
- Re-read the same file multiple times in one operation (but do on the first operation per file)
- Read files you're simply listing or exploring without changes
- Read binary or generated files unless you plan to edit them

## Example Workflow

**Correct:**
1. User asks: "Add error handling to this function in detector.py"
2. You: `read_file` → see the full detector.py
3. You: understand the function's current error handling and context
4. You: `replace_string_in_file` with precise, contextual edits

**Wrong:**
1. User asks: "Add error handling to this function in detector.py"
2. You: immediately suggest changes without reading
3. Result: suggestions conflict with imports or existing patterns

