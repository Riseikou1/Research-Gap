import {NextRequest} from "next/server";
import {afterEach, describe, expect, it, vi} from "vitest";
import {GET, POST} from "./route";

describe("same-origin backend proxy", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("carries the opaque guest cookie from creation into result retrieval", async () => {
    const upstream = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({analysis_id:"guest-result",status:"pending"}), {status:201,headers:{"content-type":"application/json","set-cookie":"research_gap_guest=signed.opaque; Path=/; HttpOnly; SameSite=Lax"}}))
      .mockResolvedValueOnce(new Response(JSON.stringify({analysis_id:"guest-result",status:"completed"}), {status:200,headers:{"content-type":"application/json"}}));
    vi.stubGlobal("fetch", upstream);

    const created = await POST(new NextRequest("http://research.test/api/backend/analyses", {method:"POST",body:JSON.stringify({research_idea:"guest idea",mode:"quick"}),headers:{"content-type":"application/json"}}), {params:Promise.resolve({path:["analyses"]})});
    expect(created.status).toBe(201);
    expect(created.headers.get("set-cookie")).toContain("research_gap_guest=signed.opaque");

    const detail = await GET(new NextRequest("http://research.test/api/backend/analyses/guest-result", {headers:{cookie:"research_gap_guest=signed.opaque"}}), {params:Promise.resolve({path:["analyses","guest-result"]})});
    expect(detail.status).toBe(200);
    expect(new Headers(upstream.mock.calls[1][1]?.headers).get("cookie")).toBe("research_gap_guest=signed.opaque");
  });
});
