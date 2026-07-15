#!/usr/bin/env node

const path = require("path");
const { pathToFileURL } = require("url");

async function main() {
  const [, , inputArg, outputArg] = process.argv;
  if (!inputArg || !outputArg) {
    console.error("Usage: node scripts/html_to_pdf.js <input.html> <output.pdf>");
    process.exit(1);
  }

  let chromium;
  try {
    chromium = require("playwright").chromium;
  } catch (error) {
    console.error("Playwright is not installed. Run: npm install --save-dev playwright && npx playwright install chromium");
    process.exit(2);
  }

  const inputPath = path.resolve(inputArg);
  const outputPath = path.resolve(outputArg);
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage();
    await page.goto(pathToFileURL(inputPath).href, { waitUntil: "networkidle" });
    await page.pdf({
      path: outputPath,
      format: "A4",
      printBackground: true,
      preferCSSPageSize: true,
      displayHeaderFooter: true,
      headerTemplate: `
        <div style="width:100%;font-size:7.5px;color:#5c6b75;padding:0 18mm;text-align:left;">
          基于深度学习的家庭电力消耗多变量时间序列预测
        </div>`,
      footerTemplate: `
        <div style="width:100%;font-size:7.5px;color:#5c6b75;padding:0 18mm;display:flex;justify-content:space-between;">
          <span>2026 年专硕机器学习课程项目报告</span>
          <span><span class="pageNumber"></span> / <span class="totalPages"></span></span>
        </div>`,
      margin: {
        top: "16mm",
        right: "16mm",
        bottom: "16mm",
        left: "16mm",
      },
    });
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(3);
});
