import Link from "next/link";
import {siteContent} from "@/lib/content";
export function Footer() { return <footer><p>© {new Date().getFullYear()} Research GAP</p><nav aria-label="Footer">
  <Link href="/privacy">Privacy</Link><Link href="/terms">Terms</Link><Link href="/pricing">Pricing</Link>
  <a href={`mailto:${siteContent.contactEmail}`}>Contact</a><Link href="/project">Project</Link>
</nav></footer>; }
