#!/usr/bin/env node
// Block commits with hardcoded absolute paths in Python files
const { execSync } = require('child_process');

const FORBIDDEN = ['/home/jovyan', '/content/', 'C:\\\\Users\\\\'];
const staged = execSync('git diff --cached --name-only --diff-filter=ACM')
  .toString().trim().split('\n').filter(f => f.endsWith('.py'));

let found = false;
for (const file of staged) {
  try {
    const content = require('fs').readFileSync(file, 'utf8');
    for (const pattern of FORBIDDEN) {
      if (content.includes(pattern)) {
        console.error(`❌ Hardcoded path "${pattern}" found in ${file}`);
        found = true;
      }
    }
  } catch (_) {}
}
process.exit(found ? 1 : 0);
