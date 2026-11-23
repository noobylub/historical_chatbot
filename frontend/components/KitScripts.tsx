"use client";

import { useEffect } from "react";

const KIT_SCRIPTS = ["/assets/system.js", "/assets/motion/motion.js", "/assets/layout/layout.js"];

/** Loads the design-kit module scripts after hydration: they rewrite parts of
 *  the DOM (nav, effects, moments), which must not race React's hydration. */
export default function KitScripts() {
  useEffect(() => {
    for (const src of KIT_SCRIPTS) {
      if (document.querySelector(`script[src="${src}"]`)) continue;
      const s = document.createElement("script");
      s.type = "module";
      s.src = src;
      document.body.appendChild(s);
    }
  }, []);
  return null;
}
