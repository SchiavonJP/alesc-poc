#!/usr/bin/env node
// Warn if a Jupyter notebook has outputs (not cleared before commit)
const fs = require('fs');
const path = require('path');

const filePath = process.env.TOOL_INPUT_FILE_PATH || '';
if (!filePath.endsWith('.ipynb')) process.exit(0);

try {
  const nb = JSON.parse(fs.readFileSync(filePath, 'utf8'));
  const hasOutputs = (nb.cells || []).some(cell =>
    cell.cell_type === 'code' && cell.outputs && cell.outputs.length > 0
  );
  if (hasOutputs) {
    console.error(
      `⚠️  Notebook outputs not cleared: ${path.basename(filePath)}\n` +
      `   Run: jupyter nbconvert --clear-output --inplace "${filePath}"`
    );
    process.exit(1);
  }
} catch (e) {
  // Not parseable — skip
}
process.exit(0);
