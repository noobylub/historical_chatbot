import type { Metadata, Viewport } from "next";
import KitScripts from "@/components/KitScripts";
import "./globals.css";

export const metadata: Metadata = {
  title: "William Morris: a conversation with the designer and socialist",
  description:
    "Chat with a model of William Morris, 1834-1896: founder of Morris & Co. and leading voice of the Arts and Crafts movement. He wakes on request; reload to begin a fresh conversation.",
  openGraph: {
    type: "website",
    title: "William Morris: a conversation with the designer and socialist",
    description:
      "Chat with a model of William Morris, 1834-1896, founder of Morris & Co. He wakes on request; reload to begin a fresh conversation.",
    images: ["/share.jpg"],
  },
  twitter: { card: "summary_large_image" },
  icons: [{ rel: "icon", url: "/favicon.svg", type: "image/svg+xml" }],
  other: { "site-kind": "software", "page-type": "software" },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: "#1c1c1c",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      data-type="software"
      data-radius="soft"
      data-ground="sky"
      data-buttons="tonal"
      data-motion=""
      suppressHydrationWarning
    >
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Figtree:wght@400..700&family=DM+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
        <link rel="stylesheet" href="/assets/system.css" />
        <link rel="stylesheet" href="/assets/ui/cards.css" />
        <link rel="stylesheet" href="/assets/motion/motion.css" />
        <link rel="stylesheet" href="/assets/backgrounds/backgrounds.css" />
        <link rel="stylesheet" href="/assets/layout/layout.css" />
        <link rel="stylesheet" href="/assets/icons/icon.css" />
        <link rel="stylesheet" href="/assets/ui/base.css" />
        <link rel="stylesheet" href="/assets/ui/chat/fragment.css" />
        {/* the dealt :root block, exactly as start.mjs printed it; after the links so it wins over system.css defaults */}
        <style
          dangerouslySetInnerHTML={{
            __html: `:root {
  --p-bg: #1c1c1c; --p-surface: #262626; --p-ink: #f1f1ef; --p-muted: #9a9b96;
  --p-accent: #8fa68e; --p-accent-ink: #1c1c1c; --p-action: #8fa68e; --p-action-ink: #1c1c1c;
  --p-inv-bg: #c8c5ba; --p-inv-surface: #bab8ae; --p-inv-ink: #1c1c1c; --p-inv-muted: #454442;
  --p-inv-accent: #1c1c1c; --p-inv-accent-ink: #c8c5ba; --p-inv-action: #1c1c1c; --p-inv-action-ink: #c8c5ba;
  --font-display: "Figtree", system-ui, sans-serif;
  --font-text: "Figtree", system-ui, sans-serif;
  --font-mono: "DM Mono", ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
  --display-weight: 500;
  --display-tracking: -0.025em;
  --display-tracking-xl: -0.035em;
}`,
          }}
        />
        {/* the hero's entrance: before first paint so the hero's words wait for motion.js */}
        <script
          dangerouslySetInnerHTML={{
            __html:
              "document.documentElement.classList.add('m-enter');setTimeout(()=>document.documentElement.classList.add('m-enter-done'),700)",
          }}
        />
        <script src="/assets/backgrounds/backgrounds.js" />
        {/* the moments guard and preloads (giant-type-pass is the dealt moment) */}
        <script
          dangerouslySetInnerHTML={{
            __html:
              "document.documentElement.classList.add('mo-js');setTimeout(()=>document.documentElement.classList.add('mo-safe'),2500)",
          }}
        />
        <link rel="modulepreload" href="/assets/moments/moments.js" />
        <link rel="modulepreload" href="/assets/moments/lib.js" />
        <link rel="modulepreload" href="/assets/moments/giant-type-pass/moment.js" />
        {/* <link rel="canonical": the client's domain is still to come */}
      </head>
      <body suppressHydrationWarning>
        {children}
        <KitScripts />
      </body>
    </html>
  );
}
