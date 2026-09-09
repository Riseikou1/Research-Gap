import Link from "next/link";
import {siteContent} from "@/lib/content";

export default function Home() { return <>
  <section className="hero"><p className="eyebrow">Evidence before assertion</p><h1>See how your research idea fits the literature.</h1>
    <p className="lede">Research GAP turns an idea into a bounded, inspectable literature analysis—showing relevant papers, evidence, coverage, and cautiously verified opportunities.</p>
    <div className="actions"><Link className="button primary" href="/analyze">Investigate an idea</Link><Link className="button secondary" href="/project">See the method</Link></div>
    <div className="limitation"><strong>Important limitation</strong><p>{siteContent.limitation}</p></div>
  </section>
  <section className="section"><p className="eyebrow">How it works</p><h2>A traceable path from question to qualification</h2><div className="steps">
    <article><b>01</b><h3>Plan the search</h3><p>Your idea becomes a bounded set of inspectable deterministic and optional terminology queries.</p></article>
    <article><b>02</b><h3>Read the evidence</h3><p>Relevant papers are ranked against your original idea, with retrieval provenance and structured evidence retained.</p></article>
    <article><b>03</b><h3>Test possible gaps</h3><p>Candidate gaps are grounded in the observed landscape and checked with targeted counterexample searches.</p></article>
  </div></section>
  <section className="section evidence-strip"><div><p className="eyebrow">Built for scrutiny</p><h2>Conclusions stay connected to papers.</h2></div><ul><li>Query and retrieval provenance</li><li>Supporting and contradicting evidence</li><li>Explicit missing data and provider failures</li><li>Abstract and full-text coverage</li></ul></section>
  <section className="section about"><p className="eyebrow">About me</p><h2>Built by a student who wanted a clearer starting point.</h2><p>{siteContent.about}</p></section>
</>; }
