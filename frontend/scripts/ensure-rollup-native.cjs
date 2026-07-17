const { execFileSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const isLinuxGlibcX64 =
  process.platform === "linux" &&
  process.arch === "x64" &&
  Boolean(process.report?.getReport?.().header?.glibcVersionRuntime);

if (!isLinuxGlibcX64) {
  process.exit(0);
}

const nativePackage = "@rollup/rollup-linux-x64-gnu";

function requireNativePackage() {
  require(nativePackage);
}

try {
  requireNativePackage();
  process.exit(0);
} catch (error) {
  if (error.code !== "MODULE_NOT_FOUND") {
    throw error;
  }
}

const rollupPackagePath = path.join(process.cwd(), "node_modules", "rollup", "package.json");
if (!fs.existsSync(rollupPackagePath)) {
  console.error("Rollup is not installed. Run npm install before starting the frontend.");
  process.exit(1);
}

const rollupVersion = JSON.parse(fs.readFileSync(rollupPackagePath, "utf8")).version;
console.log(`Installing missing Rollup native package: ${nativePackage}@${rollupVersion}`);
execFileSync(
  "npm",
  ["install", "--no-save", `${nativePackage}@${rollupVersion}`, "--no-audit", "--no-fund"],
  { stdio: "inherit" }
);

// Re-resolve from a fresh process after npm changes node_modules.
execFileSync(process.execPath, ["-e", `require(${JSON.stringify(nativePackage)})`], {
  cwd: process.cwd(),
  stdio: "inherit",
});
console.log("Rollup native package OK");
