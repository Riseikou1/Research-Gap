import type {Metadata} from "next";
import "./globals.css";
import {AuthProvider} from "@/components/auth-context";
import {Header} from "@/components/header";
import {Footer} from "@/components/footer";

export const metadata: Metadata = {title: {default: "Research GAP", template: "%s · Research GAP"}, description: "Evidence-backed investigation of possible research gaps."};
export default function RootLayout({children}: Readonly<{children: React.ReactNode}>) {
  return <html lang="en"><body><AuthProvider><Header/><main>{children}</main><Footer/></AuthProvider></body></html>;
}
