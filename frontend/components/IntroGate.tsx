"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const ENTERED_KEY = "morris.entered.v1";

/** The introduction shown before the site: who William Morris was,
 *  framed by his own floral work. Nothing behind it loads a model. */
export default function IntroGate() {
  const [open, setOpen] = useState(false);
  const buttonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    let entered = false;
    try {
      entered = localStorage.getItem(ENTERED_KEY) === "1";
    } catch {
      /* storage blocked: show the introduction */
    }
    if (!entered) setOpen(true);
  }, []);

  useEffect(() => {
    if (!open) return;
    document.documentElement.classList.add("gate-open");
    // The page behind the introduction is inert: keyboard and screen-reader
    // users stay inside the dialog until they enter.
    const behind = document.querySelectorAll<HTMLElement>("header.nav, main, footer, nav.actionbar");
    behind.forEach((el) => el.setAttribute("inert", ""));
    buttonRef.current?.focus();
    return () => {
      document.documentElement.classList.remove("gate-open");
      behind.forEach((el) => el.removeAttribute("inert"));
    };
  }, [open]);

  const enter = useCallback(() => {
    try {
      localStorage.setItem(ENTERED_KEY, "1");
    } catch {
      /* fine: the gate shows again next visit */
    }
    setOpen(false);
  }, []);

  if (!open) return null;

  return (
    <div className="gate" role="dialog" aria-modal="true" aria-labelledby="gate-h">
      <div className="gate__card">
        <div className="gate__text">
          <p className="t-label gate__label">An introduction</p>
          <h2 id="gate-h">William Morris, 1834 to 1896</h2>
          <p>
            William Morris was a British designer, writer, poet and socialist, best known as a
            leading figure in the Arts and Crafts movement.
          </p>
          <p>
            He opposed industrial mass production, believing that craftsmanship, creativity and
            beauty should be part of everyday life. He founded Morris &amp; Co., producing
            influential wallpapers, textiles and furniture.
          </p>
          <p>
            As a socialist, he advocated workers&rsquo; rights, equality and meaningful work. He
            also founded the Society for the Protection of Ancient Buildings and wrote{" "}
            <em>News from Nowhere</em>, imagining a more equal society.
          </p>
          <p>
            His central belief was that art, work and society were interconnected, and that beauty
            should be accessible to everyone, not just the wealthy.
          </p>
          <button ref={buttonRef} className="btn btn--primary" type="button" onClick={enter}>
            Enter the website
            <svg className="m-arrow" viewBox="0 0 12 12" aria-hidden="true">
              <path className="m-arrow__stem" d="M1 6h8" />
              <path className="m-arrow__head" d="M4.5 2.5 8 6l-3.5 3.5" />
            </svg>
          </button>
        </div>
        <img
          className="gate__flora"
          src="/assets/img/photo-f76003914-500.webp"
          alt=""
          width="500"
          height="446"
          data-source="stand-in"
          data-subject="Strawberry Thief by William Morris"
        />
      </div>
    </div>
  );
}
