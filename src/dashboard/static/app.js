/* Auto-PERL dashboard: client-side filtering and snapshot polling.
 *
 * Deliberately tiny and dependency-free. The campaign rows are rendered
 * server-side, so with JavaScript disabled the page still lists every
 * campaign; this only hides rows and watches for newer snapshots.
 */

(function () {
  "use strict";

  // ---------------------------------------------------------------- filters

  var table = document.querySelector("table.campaigns");
  if (table) {
    var rows = Array.prototype.slice.call(table.tBodies[0].rows);
    var controls = {
      task: document.getElementById("f-task"),
      status: document.getElementById("f-status"),
      model: document.getElementById("f-model"),
      flavor: document.getElementById("f-flavor"),
      search: document.getElementById("f-search"),
      completed: document.getElementById("f-completed"),
      archived: document.getElementById("f-archived")
    };
    var counter = document.getElementById("f-count");
    var STORAGE_KEY = "auto-perl-filters";

    function readState() {
      return {
        task: controls.task ? controls.task.value : "",
        status: controls.status ? controls.status.value : "",
        model: controls.model ? controls.model.value : "",
        flavor: controls.flavor ? controls.flavor.value : "",
        search: controls.search ? controls.search.value.toLowerCase().trim() : "",
        completed: controls.completed ? controls.completed.checked : false,
        archived: controls.archived ? controls.archived.checked : false
      };
    }

    function matches(row, state) {
      if (state.task && row.dataset.task !== state.task) return false;
      if (state.status && row.dataset.status !== state.status) return false;
      if (state.model && row.dataset.model !== state.model) return false;
      if (state.flavor && (row.dataset.flavors || "").indexOf(state.flavor) < 0) {
        return false;
      }
      if (state.completed && row.dataset.status !== "COMPLETED") return false;
      if (!state.archived && row.dataset.archived === "1") return false;
      if (state.search && (row.dataset.haystack || "").indexOf(state.search) < 0) {
        return false;
      }
      return true;
    }

    function apply() {
      var state = readState();
      var shown = 0;
      rows.forEach(function (row) {
        var visible = matches(row, state);
        row.style.display = visible ? "" : "none";
        if (visible) shown++;
      });
      if (counter) {
        counter.textContent =
          shown + (shown === 1 ? " campaign" : " campaigns") +
          (shown === rows.length ? "" : " of " + rows.length);
      }
      try {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
      } catch (err) {
        /* Private browsing or a disabled store: filtering still works. */
      }
    }

    function restore() {
      var raw;
      try {
        raw = window.localStorage.getItem(STORAGE_KEY);
      } catch (err) {
        return;
      }
      if (!raw) return;
      var state;
      try {
        state = JSON.parse(raw);
      } catch (err) {
        return;
      }
      Object.keys(controls).forEach(function (key) {
        var node = controls[key];
        if (!node || !(key in state)) return;
        // A stored task that no longer exists would hide every row, so only
        // restore values the current page actually offers.
        if (node.type === "checkbox") {
          node.checked = !!state[key];
        } else if (node.tagName === "SELECT") {
          var ok = Array.prototype.some.call(node.options, function (option) {
            return option.value === state[key];
          });
          if (ok) node.value = state[key];
        } else {
          node.value = state[key];
        }
      });
    }

    Object.keys(controls).forEach(function (key) {
      var node = controls[key];
      if (!node) return;
      node.addEventListener(node.tagName === "INPUT" && node.type === "search"
        ? "input" : "change", apply);
    });

    var reset = document.getElementById("f-reset");
    if (reset) {
      reset.addEventListener("click", function () {
        Object.keys(controls).forEach(function (key) {
          var node = controls[key];
          if (!node) return;
          if (node.type === "checkbox") node.checked = false;
          else node.value = "";
        });
        apply();
      });
    }

    restore();
    apply();
  }

  // -------------------------------------------------------------- polling

  // The site is static: while the VM is publishing, a tab left open notices
  // the new snapshot and offers a reload. It never reloads on its own, which
  // would throw away your scroll position and your filters for a number that
  // moved by one trial.
  var meta = document.querySelector('meta[name="generated-at"]');
  var pill = document.getElementById("refresh-pill");
  if (!meta || !pill) return;

  var current = meta.getAttribute("content");
  var dataUrl = document.body.getAttribute("data-index-url");
  if (!dataUrl) return;

  function poll() {
    fetch(dataUrl, { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (payload) {
        if (!payload || !payload.generated_at) return;
        if (payload.generated_at !== current) {
          pill.classList.add("visible");
        }
      })
      .catch(function () {
        /* Offline, or the Space is asleep. Try again next tick. */
      });
  }

  pill.addEventListener("click", function () {
    window.location.reload();
  });
  window.setInterval(poll, 30000);
})();
