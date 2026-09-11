"use client";

import {useEffect, useRef} from "react";

export const storySteps = [
  {number:"01", label:"Idea", title:"Start with the research question", body:"The idea is decomposed into explicit problems, methods, populations, data, outcomes, and constraints."},
  {number:"02", label:"Discovery", title:"Search from more than one angle", body:"Inspectible queries retrieve OpenAlex candidates through bounded lexical and semantic routes."},
  {number:"03", label:"Evidence", title:"Read what the source can support", body:"Selected papers become validated claims linked to title, abstract, or accessible full-text evidence."},
  {number:"04", label:"Gaps", title:"Build the observed landscape", body:"Methods, settings, datasets, findings, and limitations reveal patterns worth investigating."},
  {number:"05", label:"Verification", title:"Look deliberately for counterexamples", body:"Candidate gaps face targeted searches. Missing evidence stays uncertain; it never becomes proof of novelty."},
  {number:"06", label:"Report", title:"Keep every conclusion traceable", body:"The final report connects qualified conclusions to readable papers, evidence, and honest coverage limits."},
] as const;

export function ScrollStory() {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const scene = ref.current;
    if (!scene || !("IntersectionObserver" in window)) return;
    const steps = [...scene.querySelectorAll<HTMLElement>("[data-story-step]")];
    const observer = new IntersectionObserver(entries => {
      const visible = entries.filter(entry => entry.isIntersecting).sort((a,b) => b.intersectionRatio-a.intersectionRatio)[0];
      if (!visible) return;
      const index = visible.target.getAttribute("data-story-step") ?? "0";
      scene.dataset.activeStep = index;
      steps.forEach(step => {step.dataset.active = String(step.getAttribute("data-story-step") === index);});
    }, {rootMargin:"-28% 0px -48%", threshold:[0.15,0.45,0.75]});
    steps.forEach(step => observer.observe(step));
    return () => observer.disconnect();
  }, []);

  return <div className="scroll-story" ref={ref} data-active-step="0">
    <div className="story-visual" aria-hidden="true"><div className="story-orbit"><span>Idea</span><i/><span>Evidence</span><i/><span>Report</span></div><div className="story-card"><span>Research GAP</span><strong>Evidence before assertion.</strong><small>Bounded · traceable · qualified</small></div></div>
    <div className="story-copy">{storySteps.map((step,index)=><article key={step.number} data-story-step={index} data-active={index===0?"true":"false"}><p>{step.number} · {step.label}</p><h3>{step.title}</h3><span>{step.body}</span></article>)}</div>
  </div>;
}
