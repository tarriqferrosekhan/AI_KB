<h2>MCP Agents Demo</h2>
<b>Author:</b> <br>
<a href="https://www.linkedin.com/in/tarriq-ferrose-khan-ba527080" target="_blank"><img width="500" height="200" alt="image" src="https://github.com/user-attachments/assets/bc2bb7dc-6855-4222-b866-a80524c31089" /></a>
<br>

# Architecture

<img width="1273" height="542" alt="image" src="https://github.com/user-attachments/assets/3efe4e1e-2441-4a87-8185-6b5ade7ed5f8" />

# Components

<table>
  <tr>
    <td>MCP Servers</td>
  </tr>
  <tr>
    <td>
      <table>
        <tr>
          <td>
          <b>AuthServer.py</b><br>
          Uses https://fakestoreapi.com/auth/login to authenticate User.<br>
          <b>Tools:</b><br>
          <ul>
            <li>Authenticate</li>
            <li>UserProfile</li>
            <li>Authorise</li>
          </ul>
          </td>
        </tr>
      </table>
    </td>
  </tr>
  <tr>
    <td>
      <table>
        <tr>
          <td>
          <b>ProductServer.py</b><br>
          Uses [https://fakestoreapi.com/auth/login](https://fakestoreapi.com/products) and its variants to provide product information.<br>
          <b>Tools:</b><br>
          <ul>
            <li>GetProducts</li>
            <li>GetProductsById</li>
            <li>GetProductsByCategory</li>
            <li>GetAllCategories</li>
            <li>GetProductsBySeason</li>
            <li>GetProductsByWeather</li>
            <li>GetProductsByTitle</li>
            <li>GetProductsByDescription</li>
          </ul>
          </td>
        </tr>
      </table>
    </td>
  </tr>
</table>


# How to Launch
<table>
  <tr>
    <td>
      Auth Server
    </td>
    <td>
      Product Server
    </td>
  </tr>
  <tr>
    <td>
      py AuthServer.py<br>
    <img width="500" height="173" alt="image" src="https://github.com/user-attachments/assets/e57c3198-59c0-4553-967b-6cb805756a60" />
    </td>
    <td>
      py ProductServer.py<br>
    <img width="500" height="159" alt="image" src="https://github.com/user-attachments/assets/a49a75d5-1d43-4fc9-9b93-bd6fa724b6c2" />
    </td>
   </tr>
   <tr>
    <td>
      Weather MCP Server
    </td>
    <td>
      Launch Host Via UI.py
    </td>
  </tr>
  <tr>
    <td>
      py WeatherServer.py<br>
    <img width="500" height="171" alt="image" src="https://github.com/user-attachments/assets/bb4fbd51-ed6c-4753-a8d4-080fe17ae2f5" />
    </td>
    <td>
      py -m streamlit run UI.py<br>
    <img width="500" height="146" alt="image" src="https://github.com/user-attachments/assets/b988c834-2707-4b2c-8439-f7f3411d7579" />
    </td>
   </tr>
</table>

# UI Demo

<ol>
  <li>Initialise UI</li>
  <li>Get Weather Data</li>
  <li>Get Products Data</li>
  <li>Get Products Based on Season</li>
</ol>

<table>
  <tr>
    <td>UI Action</td>
    <td>Output Screenshot UI and Servers (Host/MCP Servers)</td>
  </tr>
  <tr>
    <td>
      <b></b>Launch UI and Initial Screen</b>
    </td>
    <td>
     <img width="596" height="538" alt="image" src="https://github.com/user-attachments/assets/6ba75a9d-f65c-4e4f-8d3f-a098798273b4" />
    </td>
  </tr>
  <tr>
    <td> <b>Get Weather Forecast</b> </td>
    <td><img width="1252" height="382" alt="image" src="https://github.com/user-attachments/assets/44fba0d3-cfd1-46ff-b397-0451be5776e3" />
 </td>
  </tr>
  <tr>
    <td><b>Get (Electronic) Products</b> </td>
    <td><img width="667" height="561" alt="image" src="https://github.com/user-attachments/assets/0ce6085d-7e93-41ab-bfe1-77fea4199d57" />
 </td>
  </tr>
  <tr>
    <td style="vertical-align:top">
      Fecthing Products From Fakestore API based on Season
      via Product MCP Server
    </td>
    <td>
      <img width="612" height="529" alt="image" src="https://github.com/user-attachments/assets/9acfb1a6-0934-4dce-8df8-426df0d64078" />
    </td>
  </tr>
</table>






