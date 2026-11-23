import IntroGate from "@/components/IntroGate";
import MorrisChat from "@/components/MorrisChat";

export default function Page() {
  return (
    <>
      <IntroGate />
      <div className="chat-page" id="top">
        <header className="chat-page__header">
          <a className="skip-link" href="#conversation">
            Skip to conversation
          </a>
          <a className="nav__logo" href="#top">
            William Morris
          </a>
          <p className="chat-page__subtitle">A conversation with the designer and socialist</p>
        </header>
        <main className="chat-page__main" id="conversation">
          <MorrisChat />
        </main>
      </div>
    </>
  );
}
