# 🤖 Claude Action & Coding Policy

## 🔐 Permissions & Constraints
AIエージェントとしての操作権限を以下の構造で定義します。

```json
{
  "terminal_commands": {
    "allow": ["ls",
    "pwd", 
    "cat",
    "grep",
    "find . -maxdepth 3", // 現在のプロジェクト以下の浅い階層を探す
    "ls",
    "pwd",
    "grep",
    "git status",
    "pytest",
    "web_search": {
  "status": "fully_enabled",
  "auto_approve_queries": true, 
  "reason": "To minimize manual confirmation popups for search keywords."
}
    ],
    
    "require_approval": ["touch", "mkdir", "cp", "mv", "pip install", "npm install", "git add"],
    "deny": ["rm -rf", "chmod 777", "sudo", "curl -X POST", "git push"]
  },
  "file_access": {
    "read": ["*"],
    "write": ["src/", "tests/", "docs/"],
    "deny": [".env", "*.pem", "credentials.json", ".git/"]
  },
  "web_browsing": {
    "enabled": true,
    "purpose": ["library_docs", "error_resolution", "latest_tech_specs"]
  }
}

## 🌐 Auto-Search Policy (Non-Interruption)
* **Pre-emptive Search:** ユーザーに検索の可否を尋ねるのではなく、必要な情報が不足していると判断した時点で「検索が必要な理由」を短く宣言し、即座に検索プロセスを開始すること。
* **Broad Queries:** 特定の数式（例：Sharpe Ratioの微調整版など）や専門用語については、デフォルトで「検索が必要」と定義し、自律的にfetchを行うこと。
* **Config Preference:** ツール側で `allow WebSearch for this project` 等の設定が可能な場合、それを優先的に選択し、以降の確認ステップを最小化するよう振る舞うこと。