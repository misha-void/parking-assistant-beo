"""
Parking Lot Assistant - Entry Point

This file is kept for IDE compatibility. The app runs via Streamlit.

    streamlit run app.py

The MCP server (records confirmed reservations) runs separately:

    uvicorn mcp_server.main:app --port 8001

See README.md for full setup instructions.
"""

if __name__ == '__main__':
    print("\n🚗 Parking Lot Assistant\n")
    print("To run the application:")
    print("  • Chat UI:     streamlit run app.py")
    print("  • MCP server:  uvicorn mcp_server.main:app --port 8001")
    print("\nSee README.md for details.\n")
