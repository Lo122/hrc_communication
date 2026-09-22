# Voice listening policy

Task prompts have separate detailed CLI text and concise spoken text in
`3_communication/message_manager.py` (`spoken=True`). CommunicationManager
accepts the short version via `speech=` and sends only that version to TTS;
without an override, it speaks the supplied message as before. The CLI and
GPT question context retain the full text. Beep and listening rules are unchanged.
Speech keeps the requested action, completion commands (`done` / `screw done`),
and deferred-start duration; command lists and task instance IDs stay on the CLI.

Voice workers report commands or one of three non-command outcomes:

| Outcome | Handling |
|---|---|
| `NO_SPEECH` | Resume listening silently; do not count a failed reply. |
| `UNRECOGNIZED` | When awaiting a reply, clarify at most once per question. During execution, adjustment, or holding, resume silently. |
| `ERROR` | Log the error and retry after `VOICE_ERROR_RETRY_SECONDS` (default 5 seconds). After two consecutive errors, announce that voice is unavailable and CLI can be used. Announce once per uninterrupted error episode. |

GPT tracks `speech_started` / `speech_stopped`. A listening timeout after speech
is logged separately from a window with no detected speech. An unsupported GPT
command after detected speech is unrecognized; without detected speech it is
ignored. Vosk empty results are ignored, while nonempty text goes through the
shared parser. Vosk has no additional local VAD in this version.

Beep is enabled for a question or clarification, not for silent retries or
background listening. `VOICE_MAX_ATTEMPTS > 1` allows one clarification; further
unrecognized replies are handled silently until a new question/state.

Worker callbacks enqueue results. `HRCSystem.process_events()` calls
`CommunicationManager.poll()` to process them on the runtime thread after task
events. Stale generations and callbacks after close are ignored. A microphone
worker that has not stopped is never replaced with another active worker.

The 8-second listening window is independent of the 20-second task response
deadline. Silent retries and clarifications do not restart that deadline.
State changes stop the previous listener and invalidate pending retries.

Non-command outcomes and reasons are written to the existing event log. No new
audio recording, wake word, model change, or VAD parameter tuning is introduced.
VAD can still mistake noise for speech; this policy reduces unnecessary prompts
but does not guarantee perfect noise rejection or command recognition.

Offline checks: `.venv/Scripts/python.exe -B -m unittest discover -s tests -v`.
The tests mock audio, network, and robot interfaces; live microphone validation
is still needed in the actual work environment.

## State-aware GPT instructions

`3_communication/voice_context.py` contains the shared base instructions, task
descriptions, and per-state templates. Edit these templates to review or refine
how GPT interprets replies. No state-dependent natural-language parsing is added
to the CLI or local command parser.

The runtime snapshots `task_instance_id`, `task_id`, and `state` from the active
task. CommunicationManager selects the template and includes the current
permission question when available. Retries and error announcements do not
replace that question. Changing task or instance invalidates the old listener
even if the robot state name stays the same.

- In `R_FREE_DRIVE`, a generic completion reply means panel adjustment is
  finished; GPT is instructed to output `done`.
- In `R_HOLDING`, the same generic reply means screwing is finished; GPT is
  instructed to output `screw done`.
- In `R_MANUAL_RECOVERY`, completion means recovery is finished: `done`.
- Explicit statements and negation override assumptions. Finishing adjustment
  alone must not be interpreted as finishing screwing.

VoiceInterface sends the generated text through `session.update.instructions`
and waits for a `session.updated` containing those instructions before beep and
recording. An already usable socket is reused: unchanged instructions need no
update, while changed instructions are updated on that socket. New connections
receive the full configuration and current instructions. Interrupted audio or
unfinished responses retain the existing close/reconnect protection.

Voice events retain the task instance ID from the listening snapshot. The
existing parser maps GPT's standard command to an event, and TaskManager still
checks event validity. The Vosk path accepts the shared listening interface but
does not use GPT instructions or reinterpret `done` by state.

To compare real speech interpretation in the standalone test (uses microphone
and the configured API credentials, but sends no robot commands):

```powershell
.venv/Scripts/python.exe 3_communication/gpt_live/gpt_stt.py --state R_FREE_DRIVE --task-id 1
.venv/Scripts/python.exe 3_communication/gpt_live/gpt_stt.py --state R_HOLDING --task-id 1
```

These run separately. Try the same phrase, such as "I'm done", in each stage.
Offline tests verify template selection, context forwarding, updates and stale
callback handling; actual model interpretation is not guaranteed by mock tests.
