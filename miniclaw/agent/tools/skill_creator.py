"""Skill creator tool."""

import argparse
from pathlib import Path


def create_skill(name: str, description: str, author: str = "", homepage: str = ""):
    """Create a new skill."""
    skill_dir = Path("miniclaw") / "skills" / "workspace" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    
    skill_file = skill_dir / "SKILL.md"
    
    content = f"""---
name: {name}
description: {description}
version: 1.0.0
author: {author}
homepage: {homepage}
metadata: {{"miniclaw":{{"emoji":"🔧","always":false,"requires":{{}}}}}}
---

# {name}

## Description

{description}

## Usage

Explain how to use this skill...

## Examples

Provide examples of usage...
"""
    
    skill_file.write_text(content, encoding="utf-8")
    print(f"Skill '{name}' created successfully at: {skill_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a new skill")
    parser.add_argument("name", help="Skill name")
    parser.add_argument("-d", "--description", required=True, help="Skill description")
    parser.add_argument("-a", "--author", default="", help="Author name")
    parser.add_argument("-hp", "--homepage", default="", help="Homepage URL")
    
    args = parser.parse_args()
    create_skill(args.name, args.description, args.author, args.homepage)