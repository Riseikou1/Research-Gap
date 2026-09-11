import React from "react";
import {fireEvent, render, screen, waitFor} from "@testing-library/react";
import {beforeEach, describe, expect, it, vi} from "vitest";
const createAnalysis=vi.fn().mockResolvedValue({analysis_id:"a1",status:"pending"});const push=vi.fn();
vi.mock("@/lib/api",()=>({createAnalysis:(...args:unknown[])=>createAnalysis(...args)}));
vi.mock("next/navigation",()=>({useRouter:()=>({push})}));
const authState:{session:{access_token:string};me:{signed_in:boolean;verified:boolean;credits:number;credit_exempt:boolean;role:string};loading:boolean}={session:{access_token:"token"},me:{signed_in:true,verified:true,credits:2,credit_exempt:false,role:"user"},loading:false};
vi.mock("@/components/auth-context",()=>({useAuth:()=>authState}));
import AnalyzePage from "./page";
describe("analysis form",()=>{
  beforeEach(()=>{createAnalysis.mockClear();push.mockClear();Object.assign(authState.me,{signed_in:true,verified:true,credits:2,credit_exempt:false,role:"user"});});
  it("propagates the full-text checkbox only for a full analysis",async()=>{render(<AnalyzePage/>);fireEvent.change(screen.getByRole("textbox",{name:/Research idea/}),{target:{value:"A sufficiently detailed research idea"}});fireEvent.click(screen.getByLabelText(/Full Gap Analysis/));fireEvent.click(screen.getByLabelText(/Use available full text/));fireEvent.click(screen.getByRole("button",{name:"Run Full Gap Analysis"}));await waitFor(()=>expect(createAnalysis).toHaveBeenCalled());expect(createAnalysis.mock.calls[0][0]).toMatchObject({mode:"full",full_text:true});expect(createAnalysis.mock.calls[0][1]).toBe("token");expect(push).toHaveBeenCalledWith("/analyses/a1")});
  it("allows a zero-credit administrator and explains real provider cost",async()=>{Object.assign(authState.me,{credits:0,credit_exempt:true,role:"admin"});render(<AnalyzePage/>);fireEvent.change(screen.getByRole("textbox",{name:/Research idea/}),{target:{value:"An administrator research idea with enough detail"}});fireEvent.click(screen.getByLabelText(/Full Gap Analysis/));expect(screen.getByText("Admin access")).toBeInTheDocument();expect(screen.getByRole("status")).toHaveTextContent(/Provider calls still have real cost/);expect(screen.queryByText(/No full-analysis credits remain/)).not.toBeInTheDocument();const submit=screen.getByRole("button",{name:"Run Full Gap Analysis"});expect(submit).toBeEnabled();fireEvent.click(submit);await waitFor(()=>expect(createAnalysis).toHaveBeenCalled());expect(createAnalysis.mock.calls[0][0]).not.toHaveProperty("is_admin");});
  it("keeps an ordinary zero-credit user blocked",()=>{Object.assign(authState.me,{credits:0,credit_exempt:false,role:"user"});render(<AnalyzePage/>);fireEvent.click(screen.getByLabelText(/Full Gap Analysis/));expect(screen.getByRole("alert")).toHaveTextContent("No full-analysis credits remain");expect(screen.getByRole("button",{name:"Run Full Gap Analysis"})).toBeDisabled();});
});
