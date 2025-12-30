from mcp.server.fastmcp import FastMCP

# サーバーの作成
mcp = FastMCP("MyServer")

@mcp.tool()
def hello(name: str) -> str:
    """Say hello to the user."""
    return f"Hello, {name}!"

if __name__ == "__main__":
    mcp.run()
    
