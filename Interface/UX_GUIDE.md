# HRC Watch — UX guide

The watch is the silent, hands-busy channel of the Communication layer. Voice
asks and the watch shows the same question, so a worker can answer when it is
too loud to speak, or check the robot's state at a glance.

## 1. Design principles

| # | Principle | How it shows up |
|---|-----------|-----------------|
| 1 | **One screen = one question** | Title ≤ 3 words ("Lift the panel?"), one supporting line. |
| 2 | **Glance in < 1 s** | The signal inside the pad says who is talking: square and stepped while the robot speaks, soft concentric rings while it listens to you. |
| 3 | **Three skins, one colour** | Deep red `#a42016` (`#e0584a` in the dark mode). **Clay** keeps the pressed pad as the subject with the drafting paper underneath; **Plate** turns the pad into one gauge face; **Sheet** lays it out as a drawing block — stamp and title on the left, the signal plotted on one wide plate. Each skin has a light and a dark mode (`?skin=clay\|plate\|sheet&mode=light\|dark`, or the buttons in the simulator). |
| 4 | **Max 1 primary + 2 secondary actions** | Big primary button bottom-centre (easy thumb reach), secondaries below. |
| 5 | **Stop is always there while the arm moves** | Red Stop in every moving state; **press-and-hold 5 s** to avoid accidental cancels with gloves, and the pill stays red for 2 s after it fires. |
| 6 | **The rail carries time** | The dimension rail under the title empties while the 20 s answer / 5 s defer runs out, and fills again with task progress while the robot moves. On the Plate skin the same value also runs around the rim. Speed is the segmented scale inside the pad, set with the −/+ beside the rail. |
| 7 | **Haptics announce, screen explains** | New question → double buzz; stop/recovery → long buzz; state change → short tap. Workers do not need to look until they feel it. |
| 8 | **The watch can never lie** | Every button is filtered through the real `StateMachine`; the server rejects anything not on screen (HTTP 409). |
| 9 | **Voice and watch are equal** | Both create the same `Event` in the same queue; the latest answer wins, the other channel updates. |

## 2. Screen map (state → screen)

| Robot state | Screen | Tone | Title | Actions |
|---|---|---|---|---|
| no task | idle | idle | Robot ready | — |
| no task, pending pool | pending | idle | *task name* + "N waiting" | Start now |
| `R_WAITING_RESPONSE` | ask | white | Lift the panel? | **Yes** · No · In 5 s (+ 20 s ring) |
| `R_DEFER` | defer | dim red | Starting soon | Cancel (+ 5 s ring) |
| `R_ACCEPTED` / `R_REDO` | starting | red | Starting… | Stop (hold) |
| `R_EXECUTING` | running | red | Lifting panel | **Pause** · Redo · Stop (hold) + speed −/+ |
| `R_PAUSED` | paused | dim red | Paused | **Resume** · Redo · Stop (hold) |
| `R_WAITING_FREE_DRIVE` | ask-free-drive | white | Guide by hand? | **Yes** · No, hold |
| `R_FREE_DRIVE` | free-drive | white | Move the panel | **Done** · Stop (hold) |
| `R_HOLDING` | holding | white | Screw it in | **Screwed** · Stop (hold) |
| `R_WAITING_HANDOVER` | ask-handover | white | Take it now? | **Yes** · Not yet |
| `R_HOLDING_HANDOVER` | holding-handover | white | Ready for it? | **Give me** · Stop (hold) |
| `R_RECOVERY_EVALUATING` | stopping | bright red | Stopping… | — |
| `R_WAITING_HOME_PERMISSION` | ask-home | bright red | Return home? | **Home** · By hand |
| `R_RETURNING_HOME` | homing | red | Going home… | — |
| `R_MANUAL_RECOVERY` | manual | white | Move arm by hand | **Done** · Abort (hold) |

The table lives in code in `watch_screens.py` — change wording there, not in JS.

## 2b. The card, the light, the pills

Every screen is one hero card — a rounded squircle with a deep red glow inside
it — plus pills underneath. Nothing else competes for attention.

| Element | Meaning | Behaviour |
|---|---|---|
| Light, glowing | the robot is working, holding, or idle | a red core in the middle of the screen fading to white at the edges |
| Card, quiet | a question is open | the glow contracts away, the card turns white, type turns dark |
| Glow pulse, slow | the robot is speaking (TTS) | `voice.speaking` |
| Glow pulse, fast | the watch is listening to you | `voice.listening` (mic open) |
| Dot row inside the card | robot speed | 9 dots; −/+ flank the card |
| Line under the dots | task progress | fills left to right; sweeps when progress is unknown |
| Big number | progress % | hidden on round watches for space |
| Seconds in the card's label | time to answer | 20 s response / 5 s defer; no frame around the screen |
| Stop pill filling with red | press-and-hold | fills over 5 s, fires, then stays red for 2 s (`HOLD_MS` / `CONFIRM_MS` in watch.js) |

Layout: the question sits above the card in uppercase; the card carries the
task label, the speed dots and the progress number. Pills: every action is a
pale translucent pill; only **Stop** is filled, in a soft red (`#c98276`). They are sized to their text, never stretched
across the screen.
Colours: `--red #a42016`, `--red-deep #6d1009`, `--red-lift #c8402f`,
ink `#2a0c07` on white.

`GET /api/watch` carries `voice: {speaking, listening}`. Live, these come from
the TTS call and the microphone mode; in `--simulate` they are faked so the
glow's pulse can be demonstrated.

## 3. Sizes (for Figma)

| Device | Frame (pt) | Safe padding | Min touch target |
|---|---|---|---|
| Apple Watch 45 mm | 198 × 242 | 11 pt sides | 34 pt high (primary 40) |
| Galaxy Watch / Wear OS round | 227 × 227 (454 px @2x) | 26 pt sides, 18 pt top | 30 pt high (primary 34) |

Type: title 20 pt semibold (17 on round), detail 10.5 pt, eyebrow 8.5 pt caps.
Pills are 38 pt high (primary) and 32 pt (secondary); on round watches 34 / 29.

## 4. How to evaluate it in the user study

* Log: the simulator's **Download CSV** writes every state change and watch press.
* Measure: answer time (trigger → watch press), wrong presses (409 toasts),
  channel used (voice vs watch, see `source` in the event log).
* Ask after each round: SUS or NASA-TLX raw + "did you notice the buzz?".
