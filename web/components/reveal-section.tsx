"use client";

import {useEffect, useRef, type ComponentPropsWithoutRef} from "react";

export function RevealSection({className = "", children, ...props}: ComponentPropsWithoutRef<"section">) {
  const ref = useRef<HTMLElement>(null);
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    element.dataset.enhanced = "true";
    element.dataset.visible = "false";
    if (!("IntersectionObserver" in window)) {
      element.dataset.visible = "true";
      return;
    }
    const observer = new IntersectionObserver(([entry]) => {
      if (!entry.isIntersecting) return;
      element.dataset.visible = "true";
      observer.disconnect();
    }, {rootMargin: "0px 0px -12%", threshold: 0.12});
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return <section ref={ref} className={`reveal-section ${className}`.trim()} {...props}>{children}</section>;
}
