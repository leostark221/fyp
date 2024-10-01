import { defaultTheme } from "@vuepress/theme-default";
import { defineUserConfig } from "vuepress/cli";
import { viteBundler } from "@vuepress/bundler-vite";

export default defineUserConfig({
  lang: "en-US",

  title: "NIDS",
  description: "Network Intrusion Detection ",

  theme: defaultTheme({
    logo: "/images/logo.png",

    navbar: ["/", "/get-started"],
  }),

  bundler: viteBundler(),
});
