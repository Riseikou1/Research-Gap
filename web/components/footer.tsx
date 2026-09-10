import Link from "next/link";
export function Footer() { return <footer><p>© {new Date().getFullYear()} Research GAP</p><nav aria-label="Footer">
  <Link href="/project">Method</Link><Link href="/pricing">Pricing</Link><Link href="/contact">Contact</Link><Link href="/privacy">Privacy</Link><Link href="/terms">Terms</Link>
</nav></footer>; }
