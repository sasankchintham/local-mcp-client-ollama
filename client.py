import asyncio
from fastmcp import Client
from ollama import chat
from fastmcp.client.transports import StdioTransport
import logging

logging.basicConfig(
    filename="mcp_raw_session.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    filemode="w"  # overwrite each run
)
logging.getLogger("fastmcp").setLevel(logging.DEBUG)

#defining the configurations
#LLM Model Config
MODEL_NAME = "llama3.2:latest"
STREAM = False

#Server config the transport is done via stdio as the server is another code file existing in same system
transport = StdioTransport(
    command="uv",
    args=["run","main.py","--log-level","DEBUG"],
    cwd=r"C:\Users\pepsi\OneDrive\Desktop\MATH-MCP-SERVER"
)
#The server is given to the client for the execution of server by client.
client = Client(transport)


#Tool discovery and communication process in an async function.
async def main():
    async with client:
        #tools discovery request
        tools = await client.list_tools()
        resources = await client.list_resources()
        ollama_tool_list=[]
        #transform the tools list to the LLM understandible format
        for tool in tools:
            tool_name=tool.name
            tool_description=tool.description
            tool_schema=tool.inputSchema
            ollama_tool_list.append({"type":"function","function":{"name":tool_name,"description":tool_description,"parameters":tool_schema}})

            #LLM Messages

        messages=[{'role':'system','content':'You are a math assistant. Use the tools when required'},
                  {'role':'user','content':'What is the product of 3 and 4.'}]
        
            #Model response
        response = chat(
            model=MODEL_NAME,
            messages=messages,
            tools=ollama_tool_list,
            stream=STREAM
        )
        message = response.message
    #Tool Call operation done by the LLM like sending the arguments and wait for the output
        if(message.tool_calls):
            tool_call = message.tool_calls[0]

            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments

            result = await client.call_tool(tool_name,tool_args)

            messages.append(message)

            messages.append({
                'role':'tool',
                "tool_name":tool_name,
                "content":str(result)
            })

            final_response = chat(
                model = MODEL_NAME,
                messages=messages,
                tools=ollama_tool_list,
                stream=STREAM
            )

            print(f"Answer by tool call:{final_response.message.content}")
        else:
            print(f" tool was not called the answer is from LLM{message.content}")

    #print(ollama_tool_list)
    #print(tools)
    #print(resources)

if __name__=="__main__":
    asyncio.run(main())

