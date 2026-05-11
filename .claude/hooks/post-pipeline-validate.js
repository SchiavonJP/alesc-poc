#!/usr/bin/env node
// After pipeline run: verify results exist and are non-empty
const fs = require('fs');

const resultsDir = 'outputs/results';
if (!fs.existsSync(resultsDir)) {
  console.error('❌ outputs/results/ does not exist — pipeline may not have run.');
  process.exit(1);
}

const files = fs.readdirSync(resultsDir).filter(f => f.endsWith('_metrics.json'));
if (files.length === 0) {
  console.error('❌ No *_metrics.json files in outputs/results/ — pipeline produced no output.');
  process.exit(1);
}

for (const f of files) {
  const content = fs.readFileSync(`${resultsDir}/${f}`, 'utf8');
  try {
    const obj = JSON.parse(content);
    if (obj.n_flagged_ensemble === undefined) {
      console.error(`⚠️  ${f} missing n_flagged_ensemble — may be malformed.`);
    }
  } catch (_) {
    console.error(`❌ ${f} is not valid JSON.`);
    process.exit(1);
  }
}
console.log(`✅ Pipeline output validated: ${files.length} category results found.`);
process.exit(0);
