# 1. Messy Data Autopsy
Find and download one government / public messy dataset
Open in Jupyter notebook
Clean missing values, inconsistent formats, broken entries
Log **every** assumption and decision in markdown cells
Finish condition: **only one Jupyter notebook**, nothing else
Fine. I’ll walk you through it slowly, like a crime scene reconstruction, not a tutorial fairy tale.

This project is **not** about cleaning data.
It’s about **learning how to think when reality is ugly**.

---

## Messy Data Autopsy: Working Phase (step by step, no nonsense)

### Phase 0: Dataset selection (don’t overthink)

Pick **one** public dataset that is obviously flawed. Government data is perfect because it’s made by committees.

Examples of what qualifies:

* Missing values everywhere
* Dates in 4 different formats
* Columns that lie
* Numbers stored as strings
* Categories spelled 6 different ways

What does *not* qualify:

* Kaggle “cleaned” datasets
* Anything with a README that solves your life

If the dataset doesn’t annoy you in the first 5 minutes, it’s too clean.

---

## Phase 1: Load and observe like a suspicious person

Open a **single Jupyter notebook**.

First cells:

* Load the data
* Print shape
* Show head and tail
* Show column names
* Basic `.info()`

Do **nothing else** yet.

Your job here is to **look**, not fix.

In a markdown cell, write:

* What this dataset claims to represent
* What looks immediately wrong or suspicious
* What columns you don’t trust yet

No fixing. No cleaning. Observation only.

---

## Phase 2: Identify mess types (this is thinking)

Now you categorize problems. Not fix them yet.

Typical categories:

* Missing values (NaN, empty strings, weird placeholders)
* Inconsistent formats (dates, units, casing)
* Broken entries (negative ages, impossible values)
* Duplicates that may or may not be real
* Columns that contradict each other

In a markdown cell:

* List each issue you find
* Point to specific columns
* Say why it’s a problem, not just that it exists

This is where your brain starts working. If this feels uncomfortable, good.

---

## Phase 3: Decide before you clean (critical)

Before touching the data, you must **decide your rules**.

For example:

* Drop rows vs impute values
* Mean vs median vs mode
* Standardize formats or discard rows
* Keep duplicates or merge them

Here’s the rule that separates amateurs from thinkers:

> **Every decision must have a reason, even if it’s imperfect.**

In markdown:

* Write each decision
* Write *why* you chose it
* Write what you’re sacrificing by choosing it

If you can’t explain the tradeoff, you’re guessing.

---

## Phase 4: Clean incrementally (small, testable steps)

Now you clean, but slowly.

One operation at a time:

* Handle missing values in one column
* Re-check stats
* Move to next issue

After each major step:

* Print shape again
* Show summary stats
* Verify you didn’t accidentally destroy the data

No giant “clean everything” function. That’s coward code.

---

## Phase 5: Validate your own work (self-suspicion)

After cleaning:

* Re-run `.info()`
* Re-check missing values
* Look for new anomalies created by your cleaning
* Compare before vs after distributions

In markdown:

* What improved?
* What uncertainty still exists?
* What assumptions might be wrong?

This is the part most people skip. That’s why their models suck later.

---

## Phase 6: Final narrative (this is the real output)

Your **final output is not the cleaned data**.

It is:

* A notebook that tells a story:

  * What the data was
  * What was wrong
  * What you assumed
  * What you changed
  * What risks remain

Your finish condition is met when:

* Someone else can read the notebook
* Disagree with your choices
* But still understand exactly why you made them

That’s professional-level thinking.

---

## What this project secretly trains you for

Without saying it out loud, you learn:

* Data reasoning
* Assumption tracking
* Complexity awareness
* Defensive thinking
* Explaining decisions under scrutiny

This is **DSA-level thinking**, just applied to reality instead of puzzles.

---

## What NOT to do (read twice)

* Don’t add ML
* Don’t visualize everything
* Don’t refactor endlessly
* Don’t create multiple notebooks
* Don’t aim for perfection

One notebook. Honest decisions. Clear thinking.

---

## Final reminder

If this project feels “simple,” you’re doing it wrong.
If it feels slightly uncomfortable and mentally tiring, you’re on target.

When you finish it, you won’t feel proud.
You’ll feel **annoyingly sharper**.

That’s the point.
