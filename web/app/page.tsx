import Link from "next/link";
import {siteContent} from "@/lib/content";

export default function Home() { return <>
  <section className="hero reveal"><p className="eyebrow">Evidence before assertion</p><h1>Find the shape of your research idea.</h1>
    <p className="lede">Research GAP turns an idea into a bounded, inspectable literature analysis—showing relevant papers, evidence, coverage, and cautiously verified opportunities.</p>
    <div className="actions"><Link className="button primary" href="/analyze">Investigate your idea</Link><Link className="button secondary" href="/project">See how it works</Link></div>
    <div className="limitation"><strong>Important limitation</strong><p>{siteContent.limitation}</p></div>
  </section>
  <section className="section narrative reveal"><div><p className="eyebrow">The problem</p><h2>A search result is not yet an explanation.</h2></div><p className="section-lede">Researchers need more than a pile of titles. They need to see what was studied, how it was evaluated, where evidence came from, and what the search could have missed.</p></section>
  <section className="section reveal"><p className="eyebrow">The process</p><h2>A traceable path from question to qualification.</h2><div className="steps">
    <article><b>01</b><h3>Frame</h3><p>Your idea becomes a bounded set of inspectable search facets and complementary queries.</p></article>
    <article><b>02</b><h3>Read</h3><p>Relevant papers are ranked, then structured evidence is retained with exact source text and provenance.</p></article>
    <article><b>03</b><h3>Verify</h3><p>Possible gaps are grounded in the observed landscape and tested with targeted counterexample searches.</p></article>
  </div></section>
  <section className="section evidence-strip reveal"><div><p className="eyebrow">The evidence</p><h2>Every important conclusion stays connected to papers.</h2><p className="section-lede">Full text is used only when open access allows it. Otherwise the report says plainly that it relied on an abstract or metadata.</p></div><ul><li>Query and retrieval provenance</li><li>Exact supporting evidence text</li><li>Counterexamples and verification notes</li><li>Abstract, metadata, and full-text coverage</li></ul></section>
  <section className="section result-story reveal"><p className="eyebrow">The result</p><h2>A calm report you can read from top to bottom.</h2><p className="section-lede">Start with a qualified verdict, move through methods and findings, then inspect candidate gaps, limitations, source coverage, and technical details only when you need them.</p><Link className="button primary" href="/analyze">Start an analysis</Link></section>
  <section className="section about reveal"><p className="eyebrow">About the project</p><h2>Built by a student who wanted a clearer starting point.</h2><p>{siteContent.about}</p></section>
</>; }
