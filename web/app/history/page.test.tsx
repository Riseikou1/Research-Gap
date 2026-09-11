import React from "react";
import {act, fireEvent, render, screen, waitFor} from "@testing-library/react";
import {beforeEach, describe, expect, it, vi} from "vitest";

const mocks = vi.hoisted(() => ({
  auth: {session: null as null | {user:{id:string};access_token:string}, sessionLoading: true},
  getHistory: vi.fn(),
}));
vi.mock("@/components/auth-context", () => ({useAuth: () => mocks.auth}));
vi.mock("@/lib/api", () => ({getHistory: (...args:unknown[]) => mocks.getHistory(...args)}));

import HistoryPage from "./page";

function deferred<T>() {let resolve!: (value:T)=>void;let reject!: (reason?:unknown)=>void;const promise=new Promise<T>((yes,no)=>{resolve=yes;reject=no});return {promise,resolve,reject}}
const record = {analysis_id:"a1",research_idea:"A saved analysis",status:"completed",mode:"full",stage:"completed",created_at:"2026-09-01T00:00:00Z",completed_at:"2026-09-01T00:01:00Z"};

describe("history loading states", () => {
  beforeEach(() => {mocks.auth.session=null;mocks.auth.sessionLoading=true;mocks.getHistory.mockReset()});

  it("never shows empty history while session restoration or the database request is pending", async () => {
    const pending = deferred<typeof record[]>();
    mocks.getHistory.mockReturnValue(pending.promise);
    const view = render(<HistoryPage/>);
    expect(screen.getByText("Restoring your session…")).toBeInTheDocument();
    expect(screen.queryByText("No analyses yet")).not.toBeInTheDocument();

    mocks.auth.sessionLoading=false;
    mocks.auth.session={user:{id:"u1"},access_token:"token-1"};
    view.rerender(<HistoryPage/>);
    expect(await screen.findByText("Loading your analyses…")).toBeInTheDocument();
    expect(screen.queryByText("No analyses yet")).not.toBeInTheDocument();

    await act(async () => pending.resolve([]));
    expect(await screen.findByText("No analyses yet")).toBeInTheDocument();
  });

  it("shows a retry state after a safe history failure", async () => {
    mocks.auth.sessionLoading=false;
    mocks.auth.session={user:{id:"u1"},access_token:"token-1"};
    mocks.getHistory.mockRejectedValueOnce(new Error("The service is temporarily unavailable."));
    mocks.getHistory.mockResolvedValueOnce([record]);
    render(<HistoryPage/>);
    expect(await screen.findByText("History could not be loaded")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", {name:"Try again"}));
    expect(await screen.findByText("A saved analysis")).toBeInTheDocument();
  });

  it("ignores a stale response after the account changes", async () => {
    const first = deferred<typeof record[]>();
    const second = deferred<typeof record[]>();
    mocks.auth.sessionLoading=false;
    mocks.auth.session={user:{id:"u1"},access_token:"token-1"};
    mocks.getHistory.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    const view = render(<HistoryPage/>);
    await waitFor(() => expect(mocks.getHistory).toHaveBeenCalledTimes(1));
    mocks.auth.session={user:{id:"u2"},access_token:"token-2"};
    view.rerender(<HistoryPage/>);
    await waitFor(() => expect(mocks.getHistory).toHaveBeenCalledTimes(2));
    await act(async () => first.resolve([{...record,research_idea:"Old account result"}]));
    expect(screen.queryByText("Old account result")).not.toBeInTheDocument();
    await act(async () => second.resolve([{...record,analysis_id:"a2",research_idea:"New account result"}]));
    expect(await screen.findByText("New account result")).toBeInTheDocument();
  });

  it("does not request private history for a signed-out visitor", () => {
    mocks.auth.sessionLoading=false;
    render(<HistoryPage/>);
    expect(screen.getByText("Sign in to view history")).toBeInTheDocument();
    expect(mocks.getHistory).not.toHaveBeenCalled();
  });
});
