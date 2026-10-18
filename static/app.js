(function () {
  var form = document.getElementById("detect-form");
  var input = document.getElementById("before");
  if (!form || !input) return;

  var counter = document.getElementById("word-count");
  var hint = document.getElementById("short-hint");
  var MIN_WORDS = 40;

  function countWords() {
    var text = input.value.trim();
    var words = text ? text.split(/\s+/).length : 0;
    counter.textContent = words + (words === 1 ? " word" : " words");
    hint.hidden = words === 0 || words >= MIN_WORDS;
  }
  input.addEventListener("input", countWords);
  countWords();

  // Block empty submissions and show progress while the model runs
  form.addEventListener("submit", function (event) {
    if (!input.value.trim()) {
      event.preventDefault();
      input.focus();
      input.setAttribute("aria-invalid", "true");
      return;
    }
    var clicked = event.submitter;
    form.querySelectorAll("button").forEach(function (button) { button.disabled = true; });
    if (clicked) {
      // Disabled buttons are not sent with the form, so pass the chosen action explicitly
      var action = document.createElement("input");
      action.type = "hidden";
      action.name = "action";
      action.value = clicked.value;
      form.appendChild(action);
      clicked.textContent = clicked.value === "humanize" ? "Rewriting…" : "Analyzing…";
    }
  });

  // The browser's Back button can restore the page with the buttons still disabled
  window.addEventListener("pageshow", function (event) {
    if (event.persisted) window.location.reload();
  });

  // Bring the result into view after a submit
  var result = document.getElementById("result");
  if (result && window.location.hash === "#detection" && input.value.trim()) {
    result.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  var copyButton = document.getElementById("copy-btn");
  if (copyButton) {
    copyButton.addEventListener("click", function () {
      var text = document.getElementById("after").value;
      var done = function () {
        copyButton.textContent = "Copied";
        setTimeout(function () { copyButton.textContent = "Copy text"; }, 1500);
      };
      if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(done);
      } else {
        document.getElementById("after").select();
        document.execCommand("copy");
        done();
      }
    });
  }
})();
