import React from "react";
import {fireEvent, render, screen, waitFor} from "@testing-library/react";
import {describe, expect, it, vi} from "vitest";
const createAnalysis=vi.fn().mockResolvedValue({analysis_id:"a1",status:"pending"});const push=vi.fn();
vi.mock("@/lib/api",()=>({createAnalysis:(...args:unknown[])=>createAnalysis(...args)}));
vi.mock("next/navigation",()=>({useRouter:()=>({push})}));
vi.mock("@/components/auth-context",()=>({useAuth:()=>({session:{access_token:"token"},me:{signed_in:true,verified:true,credits:2},loading:false})}));
import AnalyzePage from "./page";
describe("analysis form",()=>{it("propagates the full-text checkbox only for a full analysis",async()=>{render(<AnalyzePage/>);fireEvent.change(screen.getByRole("textbox",{name:/Research idea/}),{target:{value:"A sufficiently detailed research idea"}});fireEvent.click(screen.getByLabelText(/Full Gap Analysis/));fireEvent.click(screen.getByLabelText(/Use available full text/));fireEvent.click(screen.getByRole("button",{name:"Run Full Gap Analysis"}));await waitFor(()=>expect(createAnalysis).toHaveBeenCalled());expect(createAnalysis.mock.calls[0][0]).toMatchObject({mode:"full",full_text:true});expect(createAnalysis.mock.calls[0][1]).toBe("token");expect(push).toHaveBeenCalledWith("/analyses/a1")})});
