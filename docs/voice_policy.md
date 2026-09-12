# Voice listening policy

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
