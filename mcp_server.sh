#!/bin/bash
cd /Users/sasha/IdeaProjects/personal_projects/ohmyrepos
export PYTHONPATH=/Users/sasha/IdeaProjects/personal_projects/ohmyrepos
exec /Users/sasha/miniconda3/bin/python3 mcp_server.py 2>/tmp/ohmyrepos_mcp.log
