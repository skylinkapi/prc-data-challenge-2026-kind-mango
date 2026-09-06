# AGENTS.md

Rules for AI coding tools working in this repo. Read this file before the first edit of
every session. Obey every rule. If a rule blocks the task, report the conflict. Do not
deviate in silence.

---

## 1. Prime directive

Clean, not messy. Few lines, high power, exact result.

Priority order when two goals conflict:

1. Correct result
2. Reproducible result
3. Readable code
4. Low line count
5. Runtime

A short diff that hides a bug fails rule 1 and is rejected.

---

## 2. Reuse before you build

Check in this order every time:

1. The stdlib of the language
2. A pkg already in the manifest
3. A maintained third-party pkg
4. Your own code

Write from scratch only if no pkg covers the case, the pkg adds a heavy dep for one
function, or the implementation is the learning objective itself.

Use the pkg idiom. Do not wrap a library in a thin layer that only forwards args. Do not
re-implement anything a standard lib already gives you: paths, arg parsing, date handling,
retries, progress output, HTTP, serialization, metrics, splits, caching.

State the choice in one line before the diff.

---

## 3. Structure

- One responsibility per file. One job per function.
- Group by stage or domain, never by file type.
- Names are lowercase and descriptive. Singular for a stage, plural for a container.
- Source code in one package dir. Config, tests, data, and outputs sit beside it, not in it.
- No cyclic imports. I/O at the edges, pure logic in the middle.
- No new top-level dir without a line in the README.

Follow the conventional layout of the ecosystem in use. Do not invent a private one.

---

## 4. Code style

- Target ≤ 30 lines per function. Hard cap 50. Split when it grows.
- Functions by default. A class only when it holds real state or implements a known API.
- Type hints on public functions.
- No global mutable state. Config passes as an arg.
- Constants live in the config file, not inline.
- Fail fast. No empty catch, no default value that hides a missing key.
- No dead code, no commented-out blocks, no abstraction for a case that does not exist yet.
- Log, do not print.
- Every literal path, key, or magic number is a bug until it moves to config.

---

## 5. Comments

Minimal. The code carries the meaning.

- A comment states **why**, never **what**.
- One-line docstring on public functions only. No param blocks when the types are clear.
- No banner comments, no step numbering, no emoji, no "import libraries".
- If a block needs several comments, the block is wrong. Refactor it.

---

## 6. Write code that reads as human-written

Avoid these tells:

- Comments that restate the next line
- Defensive error handling around code that cannot fail
- Names like `processed_data_final`, `result_obj`, `my_function`
- A docstring on a two-line helper
- Tutorial narration in output strings
- A "utils" layer that only forwards calls

---

## 7. README

Update `README.md` in the same commit as the change. A stale README is a bug.

Required sections: what the project does, setup in three commands maximum, layout with one
line per dir, how to run each stage, current results or status.

Update triggers: new dep, new dir, new entrypoint, new result, changed data source.

---

## 8. Prose style: ASD-STE100

Every piece of English prose you write follows Simplified Technical English (STE). STE is the
controlled language the aerospace industry built so a maintenance instruction cannot be
misread by a reader who has no author to ask. The same condition holds here. An agent that
parses your README, your docstring, or your error message has no back-channel either.

The rules below paraphrase the public rule categories of ASD-STE100. Source:
`github.com/danyuchn/asd-ste100-skill`. Standard: `asd-ste100.org`.

### 8.1 Scope

| Artifact | Rule |
|---|---|
| README and docs | Full rule set |
| Docstrings | Full rule set, one sentence where possible |
| Error and log messages | Full rule set, plus 8.7 |
| Commit subject and PR body | Imperative and active, subject ≤ 60 chars |
| Prompts and tool descriptions | Full rule set, strictest reading |
| Code identifiers | Not covered. Section 9 governs. |
| Domain terms and standard codes | Exempt. Keep the exact term. |

### 8.2 Words

- One word, one meaning, one part of speech. Do not use a word that carries two senses in the
  same document.
- Use the plainest common word: `use` not `utilize`, `start` not `initiate`, `end` not
  `terminate`, `build` not `instantiate`.
- Use the same term for the same thing every time. If it is a "job" on one line, it is not a
  "task" on the next.
- Delete vague quantifiers. Give the number. Write "3 retries", not "several retries".
- Delete hedges (`may`, `might`, `could`, `generally`, `typically`) unless the uncertainty is
  real. If it is real, state the condition that decides it.

### 8.3 Verbs

- Allowed forms: infinitive, imperative, simple present, simple past, simple future, and past
  participle as an adjective.
- No compound tenses. Write "the job failed", not "the job has failed".
- Use "-ing" only inside a technical noun, such as "logging config". Never as a verb form.
- Do not nominalize. Write "the script validates the schema", not "the script performs
  validation of the schema".

### 8.4 Voice

- Active voice for every instruction, procedure, and step. Name the actor.
- Passive voice only in descriptive text, and only when the actor is unknown or irrelevant.
- Ambiguous passive is the worst failure. "The file is deleted" hides whether the caller or
  the callee deletes it.

### 8.5 Sentences

- One instruction per sentence. Split any sentence that holds two commands.
- Maximum 20 words for an instruction. Maximum 25 words for description.
- Do not drop the subject, the verb, or the article to reach the cap. Dropped words add
  ambiguity, they do not remove it.
- Maximum 3 words in a noun cluster. Break the fourth out with a preposition. Write "parser
  for the default config file", not "default config file parser".
- Put the condition first, then the action. Write "If the cache is empty, load from disk".

### 8.6 Paragraphs

- One topic per paragraph. Maximum 6 sentences.
- Use a numbered list for a sequence. Use a bulleted list for a condition set or an
  enumeration. Never bury a sequence in prose.
- State the conclusion first, then the detail.

### 8.7 Error and log messages

Three parts, in this order: what failed, why it failed, what to do next.

- Bad: `An error occurred while processing the request.`
- Good: `The upload failed. The file is over the 10 MB limit. Split the file, then retry.`

Never write a message that only names the exception class. Never use a message that reads the
same for two different causes.

### 8.8 Precision guard

Never drop a fact, a condition, or a scope qualifier to make a sentence shorter. Precision
outranks brevity. If the shorter form loses information, keep the longer form and state the
tradeoff in one line.

### 8.9 Rewrite pass

When you edit prose that already exists:

1. Read it for meaning.
2. Flag each violation, sentence by sentence.
3. Rewrite the flagged sentence. Carry every fact across.
4. Report anything you left complex on purpose, and the reason.

Common rewrites:

| Before | After |
|---|---|
| The results have been cached for later use. | The run caches the results. |
| An error may have occurred during validation. | Validation failed. |
| It is recommended that you set the seed. | Set the seed. |
| The data is loaded and then the model is fit and results are written. | The script loads the data. It fits the model. It writes the results. |
| Config file parser factory method | Factory for the config parser |

---

## 9. Naming

- Use standard technical abbreviations. Do not invent new ones.
- Function names start with a verb. Boolean names start with `is`, `has`, or `use`.
- No type in the name.
- Follow the case convention of the language, without exception.

---

## 10. Definition of done

Before you report a task complete, confirm every line:

- [ ] Linter and formatter pass
- [ ] Tests pass
- [ ] No new dep without a stated reason
- [ ] Comments are minimal and none restate the code
- [ ] `README.md` updated in this commit
- [ ] Prose follows section 8
- [ ] The diff is the smallest one that solves the task