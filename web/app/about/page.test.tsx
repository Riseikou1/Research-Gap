import React from "react";
import {render, screen} from "@testing-library/react";
import {describe, expect, it} from "vitest";
import AboutPage from "./page";

describe("about page", () => {
  it("explains the full flow, uncertainty, limitations, and independent attribution", () => {
    render(<AboutPage/>);
    expect(screen.getByRole("heading", {name:/clearer starting point/i})).toBeInTheDocument();
    expect(screen.getByText("Research idea decomposition")).toBeInTheDocument();
    expect(screen.getByText("Targeted verification and counterexample search")).toBeInTheDocument();
    expect(screen.getByText((_, element) =>
      element?.tagName === "P" && Boolean(element.textContent?.includes("Uncertain verdict means")),
    )).toBeInTheDocument();
    expect(screen.getByText(/does not prove global novelty/i)).toBeInTheDocument();
    expect(screen.getByText(/independent project/i)).toBeInTheDocument();
    expect(screen.getByRole("link", {name:"OpenAlex"})).toHaveAttribute("href", "https://openalex.org/");
    expect(screen.getByRole("link", {name:"OpenAI"})).toHaveAttribute("href", "https://openai.com/");
  });
});
