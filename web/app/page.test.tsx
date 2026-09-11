import {readFileSync} from "node:fs";
import React from "react";
import {render, screen} from "@testing-library/react";
import {describe, expect, it} from "vitest";
import Home from "./page";

describe("home", () => {
  it("renders the scrollytelling project story and links to analyze without a search form", () => {
    const {container} = render(<Home/>);
    expect(screen.getByText(/I’m Temuujin/)).toBeInTheDocument();
    expect(screen.getAllByRole("link", {name:/Investigate your idea/})[0]).toHaveAttribute("href", "/analyze");
    expect(screen.getAllByRole("link", {name:/See how Research GAP works/})[0]).toHaveAttribute("href", "/about");
    expect(screen.getByText("The problem")).toBeInTheDocument();
    expect(screen.getByText("The result")).toBeInTheDocument();
    expect(container.querySelectorAll("[data-story-step]")).toHaveLength(6);
    expect(container.querySelector(".story-visual")).toBeInTheDocument();
    expect(container.querySelector("form")).toBeNull();
  });

  it("keeps the story visible without motion and returns to natural flow on smaller screens", () => {
    const css = readFileSync("app/globals.css", "utf8");
    expect(css).toContain("@media (prefers-reduced-motion: reduce)");
    expect(css).toMatch(/prefers-reduced-motion[\s\S]*\.reveal-section\[data-enhanced="true"\]\s*\{\s*opacity:\s*1;\s*transform:\s*none;/);
    expect(css).toMatch(/@media \(max-width: 900px\)[\s\S]*\.story-visual\s*\{\s*position:\s*relative;/);
  });
});
