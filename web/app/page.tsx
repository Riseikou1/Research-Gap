import Link from "next/link";
import {RevealSection} from "@/components/reveal-section";
import {ScrollStory} from "@/components/scroll-story";
import {siteContent} from "@/lib/content";

export default function Home() { return <>
  <section className="hero home-hero"><div className="hero-glow" aria-hidden="true"/><p className="eyebrow">Evidence before assertion</p><h1>Find the shape of your research idea.</h1>
    <p className="lede">Research GAP turns an idea into a bounded, inspectable literature analysis—showing relevant papers, validated evidence, coverage, and cautiously verified opportunities.</p>
    <div className="actions"><Link className="button primary" href="/analyze">Investigate your idea</Link><Link className="button secondary" href="/about">See how Research GAP works</Link></div>
    <div className="limitation"><strong>Important limitation</strong><p>{siteContent.limitation}</p></div>
  </section>
  <RevealSection className="section narrative"><div><p className="eyebrow">The problem</p><h2>A search result is not yet an explanation.</h2></div><p className="section-lede">Researchers need more than a pile of titles. They need to see what was studied, how it was evaluated, where evidence came from, and what the search could have missed.</p></RevealSection>
  <RevealSection className="story-section"><div className="story-intro"><p className="eyebrow">From idea to report</p><h2>A deliberate research story.</h2><p className="section-lede">Scroll through the stages. The interface never takes control of scrolling; each step simply brings the next part of the process into focus.</p></div><ScrollStory/></RevealSection>
  <RevealSection className="section evidence-strip"><div><p className="eyebrow">The evidence</p><h2>Every important conclusion stays connected to papers.</h2><p className="section-lede">Full text is used only when open access allows it. Otherwise the report says plainly that it relied on an abstract or metadata.</p></div><ul><li>Query and retrieval provenance</li><li>Exact supporting evidence text</li><li>Counterexamples and verification notes</li><li>Abstract, metadata, and full-text coverage</li></ul></RevealSection>
  <RevealSection className="section result-story"><p className="eyebrow">The result</p><h2>A calm report you can read from top to bottom.</h2><p className="section-lede">Start with a qualified verdict, move through validated methods and findings, then inspect candidate gaps, source coverage, and technical details only when you need them.</p><Link className="button primary" href="/analyze">Start an analysis</Link></RevealSection>
  <RevealSection className="section about"><p className="eyebrow">About the project</p><h2>Built by a student who wanted a clearer starting point.</h2><p>{siteContent.about}</p><Link className="text-link" href="/about">See how Research GAP works <span aria-hidden="true">→</span></Link></RevealSection>
  <RevealSection className="final-cta"><p className="eyebrow">Your next question</p><h2>Begin with an idea. Leave with an evidence trail.</h2><Link className="button primary" href="/analyze">Investigate your idea</Link></RevealSection>
</>; }
