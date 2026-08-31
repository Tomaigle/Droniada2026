---
description: "Use when developing features or solving tasks for Droniada Challenge 2026. Follow safety principles and ensure deep understanding of the core challenge before proposing solutions."
applyTo: "**"
---
# Droniada Challenge 2026 — Core Principles

Reference: `dokumentacja/Regulamin_konkursu_Droniada_Challenge_2026.pdf`

## Core Principle: Safety First

All code, algorithms, and systems must prioritize **safety** above all other considerations:

- Fail-safe defaults and error handling
- No risky operations without explicit safeguards
- Test edge cases and failure modes
- Document safety constraints clearly

## Core Principle: Deep Understanding of the Challenge

Before proposing any solution:

1. **Understand the actual problem** — What is the core challenge we're solving?
2. **Identify edge cases** — What makes this problem difficult?
3. **Propose practical solutions** — Think about implementation requirements, testing needs, and real-world constraints

### Example: Solar Panel Anomaly Detection (Ogień i Woda Basic)

**Challenge:** Detect solar panel anomalies in images

**Right approach:**

- Understand what "anomaly" means in the context
- Propose a **test data generator script** to create synthetic panel images for training
- Test detection accuracy on both generated and real panels
- Document known limitations

**Wrong approach:**

- Jump straight to model training without understanding the data
- Use only real panels without generating test cases
- Assume anomalies are obvious without exploring edge cases

## When Proposing Solutions

✅ **Do:**

- Articulate what the core challenge is
- Explain why your approach solves it
- Propose test/validation strategies (especially test data generators where needed)
- Consider safety implications

❌ **Don't:**

- Assume you understand the problem without clarifying
- Skip test planning and validation
- Ignore edge cases or failure modes
- Implement without safety considerations

## Tasks in This Project

- **LastMileBasic** — Gripper and delivery logistics
- **maly_dron** — Small drone telemetry and control
- **ogien_i_woda_basic** — Fire/water detection (solar panels, audio classification)
- **ogien_i_woda_advanced** — Advanced sound and image classification
- **realsense** — RealSense camera integration for detection

For each task, ensure you understand the *core challenge* before coding.
