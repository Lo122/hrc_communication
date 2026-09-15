const clock = document.querySelector("#clock");
const feedback = document.querySelector("#feedback");
const permissionTitle = document.querySelector("#permission-title");
const permissionDetail = document.querySelector("#permission-detail");
const permissionActions = document.querySelector(".permission-actions");
const commandButtons = [...document.querySelectorAll("[data-command]")];

const feedbackCopy = {
  H_ACCEPT: "Start response sent",
  H_REFUSE: "No response sent",
  H_DEFER: "Later response sent",
};

const state = {
  taskInstanceId: null,
  sending: false,
  submitted: false,
  pollTimer: null,
};

function setButtonsDisabled(disabled) {
  commandButtons.forEach((button) => {
    button.disabled = disabled;
  });
}

function schedulePermissionRefresh(delay = 750) {
  window.clearTimeout(state.pollTimer);
  if (!state.submitted) {
    state.pollTimer = window.setTimeout(refreshPermission, delay);
  }
}

async function refreshPermission() {
  if (state.sending || state.submitted) {
    return;
  }

  try {
    const response = await fetch("/api/permission", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Permission state failed: ${response.status}`);
    }

    const permission = await response.json();
    if (permission.available) {
      state.taskInstanceId = permission.task_instance_id;
      permissionTitle.textContent = permission.question;
      permissionDetail.textContent = "Robot task R1 is ready.";
      feedback.classList.remove("is-visible");
      setButtonsDisabled(false);
    } else {
      state.taskInstanceId = null;
      permissionDetail.textContent = "Waiting for H0…";
      feedback.textContent = "No permission request yet";
      feedback.classList.add("is-visible");
      setButtonsDisabled(true);
    }
  } catch (error) {
    console.error(error);
    state.taskInstanceId = null;
    permissionDetail.textContent = "Backend unavailable";
    feedback.textContent = "Connecting…";
    feedback.classList.add("is-visible");
    setButtonsDisabled(true);
  }

  schedulePermissionRefresh();
}

async function selectResponse(button) {
  if (!state.taskInstanceId) {
    return;
  }

  const command = button.dataset.command;
  state.sending = true;
  setButtonsDisabled(true);
  feedback.textContent = "Sending…";
  feedback.classList.add("is-visible");

  try {
    const response = await fetch("/api/permission", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        command,
        task_instance_id: state.taskInstanceId,
      }),
    });

    if (!response.ok) {
      throw new Error(`Permission response failed: ${response.status}`);
    }

    const result = await response.json();
    state.submitted = true;
    window.dispatchEvent(
      new CustomEvent("hrc-permission-response", { detail: result.event }),
    );
    console.info("HRC permission response", result.event);
  } catch (error) {
    console.error(error);
    state.taskInstanceId = null;
    state.sending = false;
    feedback.textContent = "Could not send · checking task";
    schedulePermissionRefresh(300);
    return;
  }

  permissionActions.classList.add("has-response");
  commandButtons.forEach((candidate) => {
    const selected = candidate === button;
    candidate.classList.toggle("is-selected", selected);
    candidate.setAttribute("aria-pressed", String(selected));
  });
  feedback.textContent = feedbackCopy[command];
}

permissionActions.addEventListener("click", (event) => {
  const button = event.target.closest("[data-command]");
  if (!button || button.disabled || state.sending || state.submitted) {
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
refreshPermission();
window.setInterval(updateClock, 1000);
