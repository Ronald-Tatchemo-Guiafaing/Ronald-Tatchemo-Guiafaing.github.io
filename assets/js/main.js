(function () {
  const panels = document.querySelectorAll(".panel");
  const navLinks = document.querySelectorAll(".side-nav a[data-panel]");
  const gotoBtns = document.querySelectorAll("[data-goto]");

  function showPanel(id) {
    panels.forEach((p) => p.classList.toggle("active", p.id === "panel-" + id));
    navLinks.forEach((a) => a.classList.toggle("active", a.dataset.panel === id));
    document.body.classList.toggle("panel-home", id === "home");
    const active = document.getElementById("panel-" + id);
    if (active) active.scrollTop = 0;
  }

  document.body.classList.add("panel-home");

  navLinks.forEach((a) => {
    a.addEventListener("click", (e) => {
      e.preventDefault();
      showPanel(a.dataset.panel);
    });
  });

  gotoBtns.forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      showPanel(btn.dataset.goto);
    });
  });

  document.addEventListener("keydown", (e) => {
    const order = ["home", "about", "resume", "portfolio", "contact"];
    const current = document.querySelector(".side-nav a.active")?.dataset.panel;
    const idx = order.indexOf(current);
    if (e.key === "ArrowDown" && idx < order.length - 1) showPanel(order[idx + 1]);
    if (e.key === "ArrowUp" && idx > 0) showPanel(order[idx - 1]);
  });
})();
