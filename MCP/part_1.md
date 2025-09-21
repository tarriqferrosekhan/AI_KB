<h2>Where Exactly MCP is adding Value? - A Comprehensive comparitive study</h2>
Author:<br>
<img width="500" height="200" alt="image" src="https://github.com/user-attachments/assets/bc2bb7dc-6855-4222-b866-a80524c31089" />

<h2>References:</h2> 
<ul>
<li><a href="https://modelcontextprotocol.io/docs/getting-started/intro" target="_blank">MCP Introduction</a></li>
<li><a href="https://modelcontextprotocol.io/docs/learn/architecture" target="_blank">MCP Architecture</a></li>
<li><a href="https://modelcontextprotocol.io/docs/develop/build-server" target="_blank">Build a MCP Server</a></li>
<li><a href="https://modelcontextprotocol.io/docs/develop/build-client" target="_blank">Build a MCP Client</a></li>
</ul>
<h2>Comparitive Study with Code Examples</h2>
<ul>
  <li>Wrote an MCP Server and Client for Weather API example from modelcontextprotocol.io <b>but using OPEN AI </b>, instead of <b>Claude Desktop</b></li>
  <li>Compared the MCP design and implementation with other methods - LLM+NO MCP and classical REST API <b>to enlighten where Exactly MCP is adding Value</b></li>
</ul>
  <table>
    <thead>
      <tr>
        <th>Aspect</th>
        <th>Classical REST API</th>
        <th>LLM + Custom Wrapper (No MCP)</th>
        <th>MCP (Model Context Protocol)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Weather Fore cast + Alert Example <b>CODE</b></td>
        <td><a href="https://github.com/tarriqferrosekhan/AI_DEV/tree/main/03_mcp/Weather.RESTAPI">Weather.RESTAPI</a></td>
        <td><a href="https://github.com/tarriqferrosekhan/AI_DEV/tree/main/03_mcp/Weather.OpenAI.NOMCP">Weather.OpenAI.NOMCP</a></td>
        <td><a href="https://github.com/tarriqferrosekhan/AI_DEV/tree/main/03_mcp/Weather.OpenAI.MCP">Weather.OpenAI.MCP</a></td>
      </tr>
      <tr>
        <td>Process Flow Comparison</td>
        <td>
          <!--img width="300" height="350" alt="image" src="https://github.com/user-attachments/assets/02fd486d-b5d1-42c0-aaf6-5d805b4b84c5" /-->
          <img width="300" height="350" alt="image" src="https://github.com/user-attachments/assets/a26f4a17-4bc7-4c87-8bb1-f495b33a41b2" />
        </td>
        <td><img width="300" height="350" alt="image" src="https://github.com/user-attachments/assets/352a0e1f-2ce7-4264-ab4e-bd133dfe3e44" />
</td>
        <td><img width="300" height="350" alt="image" src="https://github.com/user-attachments/assets/c34f56ce-1d42-43a8-91b2-d894e996125d"/></td>
      </tr>
      <tr>
        <td><strong>Discovery</strong></td>
        <td>OpenAPI/Swagger docs, developer reads them</td>
        <td>Developer manually defines schemas &amp; wrappers</td>
        <td>Client calls <code>list_tools</code> → server advertises tools dynamically</td>
      </tr>
      <tr>
        <td><strong>Integration effort</strong></td>
        <td>Client must know endpoints (<code>/forecast</code>, <code>/alerts</code>)</td>
        <td>Developer writes glue code (<code>get_forecast</code>, <code>get_alerts</code>)</td>
        <td>Zero glue code: tools + schemas are self-described</td>
      </tr>
      <tr>
        <td><strong>Transport</strong></td>
        <td>HTTP(S)</td>
        <td>HTTP(S) via wrapper</td>
        <td>JSON-RPC (stdio, WebSocket, SSE)</td>
      </tr>
      <tr>
        <td><strong>Schema</strong></td>
        <td>Defined in OpenAPI</td>
        <td>Manually replicated in client code</td>
        <td>JSON Schema returned by server</td>
      </tr>
      <tr>
        <td><strong>User → App Flow</strong></td>
        <td>User → App → Server → Weather API</td>
        <td>User → LLM → Custom Wrapper → Weather API</td>
        <td>User → LLM → MCP Client → MCP Server → Weather API</td>
      </tr>
      <tr>
        <td><strong>Maintainability</strong></td>
        <td>Requires versioning &amp; contract updates</td>
        <td>Wrapper &amp; schema break when API changes</td>
        <td>Server advertises tools, client just re-fetches list</td>
      </tr>
      <tr>
        <td><strong>AI-native</strong></td>
        <td>❌ Designed for humans/devs</td>
        <td>⚠️ Manually adapted for LLMs</td>
        <td>✅ Built for LLMs &amp; agents</td>
      </tr>
    </tbody>
  </table>

  More to come on MCP
  Happy Learning
  Author - <a href="www.linkedin.com/in/tarriq-ferrose-khan-ba527080" target="_blank">Tarriq Ferrose Khan</a>


