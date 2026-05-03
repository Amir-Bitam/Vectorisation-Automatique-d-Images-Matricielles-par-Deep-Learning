import React from "react";
import { Wand2 } from "lucide-react";

function NavBar() {
  return (
    <header className="mx-auto flex w-full max-w-7xl items-center justify-between px-5 py-5 sm:px-8">
      <div className="flex min-w-0 items-center gap-3">
        <div className="grid h-10 w-10 shrink-0 place-items-center rounded-lg bg-gradient-to-br from-blue-600 to-violet-600 text-white shadow-lg shadow-blue-600/20">
          <Wand2 className="h-5 w-5" aria-hidden="true" />
        </div>
        <span className="truncate text-base font-bold text-slate-950 sm:text-lg">PFE Vectorization App</span>
      </div>
      <nav className="hidden items-center gap-7 text-sm font-semibold text-slate-600 sm:flex">
        <a className="transition hover:text-blue-700" href="#api">
          API
        </a>
        <a className="transition hover:text-blue-700" href="#settings">
          Settings
        </a>
        <a className="transition hover:text-blue-700" href="#about">
          About
        </a>
      </nav>
    </header>
  );
}

export default NavBar;
