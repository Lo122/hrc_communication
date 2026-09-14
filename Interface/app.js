const clock = document.querySelector("#clock");
const feedback = document.querySelector("#feedback");
const permissionActions = document.querySelector(".permission-actions");
const commandButtons = [...document.querySelectorAll("[data-command]")];

const feedbackCopy = {
  H_ACCEPT: "Start response sent",
  H_REFUSE: "No response sent",
  H_DEFER: "Later response sent",
};

function setButtonsDisabled(disabled) {
  commandButtons.forEach((button) => {
    button.disabled = disabled;
  });
}

async function selectResponse(button) {
  const command = button.dataset.command;

  setButtonsDisabled(true);
  feedback.textContent = "Sending…";
  feedback.classList.add("is-visible");

  try {
    const response = await fetch("/api/permission", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command }),
    });

    if (!response.ok) {
      throw new Error(`Permission response failed: ${response.status}`);
    }

    const result = await response.json();
    window.dispatchEvent(new CustomEvent("hrc-permission-response", { detail: result.event }));
    console.info("HRC permission response", result.event);
  } catch (error) {
    console.error(error);
    setButtonsDisabled(false);
    feedback.textContent = "Could not send · try again";
    return;
  }

  permissionActions.classList.add("has-response");
  commandButtons.forEach((candidate) => {
    const selected = candidate === button;
    candidate.classList.toggle("is-selected", selected);
    candidate.setAttribute("aria-pressed", String(selected));
  });

  feedback.textContent = feedbackCopy[command];
  feedback.classList.add("is-visible");
}

permissionActions.addEventListener("click", (event) => {
  const button = event.target.closest("[data-command]");
  if (!button || button.disabled) {
    return;
  }
  void selectResponse(button);
});

function updateClock() {
  const now = new Date();
  clock.dateTime = now.toISOString();
  clock.textContent = now.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

updateClock();
window.setInterval(updateClock, 1000);
