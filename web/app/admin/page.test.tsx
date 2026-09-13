import React from "react";
import {render, screen} from "@testing-library/react";
import {afterEach, describe, expect, it, vi} from "vitest";

vi.mock("@/components/auth-context", () => ({useAuth: () => ({
  session:{access_token:"test-token"}, me:{role:"admin"}, loading:false,
})}));

import Admin from "./page";

afterEach(() => vi.restoreAllMocks());

describe("admin operational summary", () => {
  it("renders provider usage, budget, and safe recent failures only for admins", async () => {
    vi.stubGlobal("fetch", vi.fn().mockImplementation((url:string) => {
      const body = url.endsWith("/admin/users") ? [] : {
        user_count:1, analyses:[], failed_payment_events:[], audit_log:[],
        operations:{
          provider_usage_today:{request_count:2,input_tokens:120,output_tokens:30,total_tokens:150,cost_usd:.001,unavailable_count:1},
          provider_budget:{enabled:true,daily_limit_usd:5,committed_usd:.25,reservation_usd:.25},
          recent_failures:[{stage:"analysis_job",category:"provider_failure"}],
        },
      };
      return Promise.resolve(new Response(JSON.stringify(body), {status:200}));
    }));
    render(<Admin/>);
    expect(await screen.findByRole("heading", {name:"Operational diagnostics"})).toBeInTheDocument();
    expect(await screen.findByText(/provider_usage_today/)).toBeInTheDocument();
    expect(screen.getByText(/unavailable_count/)).toBeInTheDocument();
    expect(screen.queryByText(/test-token/)).not.toBeInTheDocument();
  });
});
