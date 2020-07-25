<template>
  <v-simple-table dark>
    <template v-slot:default>
      <thead>
        <tr>
          <th class="text-left">INDEX</th>
          <th class="text-left">COUNTRY</th>
          <th class="text-left">CONFIRMED</th>
          <th class="text-left">RECOVERED</th>
          <th class="text-left">DEATH</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>1</td>
          <td>{{ japan[japan.length-1].Country }}</td>
          <td>{{ japan[japan.length-1].Confirmed}}</td>
          <td>{{ japan[japan.length-1].Recovered}}</td>
          <td>{{ japan[japan.length-1].Deaths}}</td>
        </tr>
        <tr>
          <td>2</td>
          <td>{{ austraila[austraila.length-1].Country }}</td>
          <td>{{ austraila[austraila.length-1].Confirmed}}</td>
          <td>{{ austraila[austraila.length-1].Recovered}}</td>
          <td>{{ austraila[austraila.length-1].Deaths}}</td>
        </tr>
        <tr>
          <td>3</td>
          <td>{{ newzealand[newzealand.length-1].Country }}</td>
          <td>{{ newzealand[newzealand.length-1].Confirmed}}</td>
          <td>{{ newzealand[newzealand.length-1].Recovered}}</td>
          <td>{{ newzealand[newzealand.length-1].Deaths}}</td>
        </tr>
        <tr>
          <td>4</td>
          <td>{{ uk[uk.length-1].Country }}</td>
          <td>{{ uk[uk.length-1].Confirmed}}</td>
          <td>{{ uk[uk.length-1].Recovered}}</td>
          <td>{{ uk[uk.length-1].Deaths}}</td>
        </tr>
        <tr>
          <td>5</td>
          <td>{{ usa[usa.length-1].Country }}</td>
          <td>{{ usa[usa.length-1].Confirmed}}</td>
          <td>{{ usa[usa.length-1].Recovered}}</td>
          <td>{{ usa[usa.length-1].Deaths}}</td>
        </tr>
      </tbody>
    </template>
  </v-simple-table>
</template>

<script>
import axios from 'axios';
export default {
    name:'Dashboard',
    data() {
    return {
      japan:[],
      uk:[],
      newzealand:[],
      austraila:[],
      usa:[],
      errors: []
      }
    },
    methods:{

    },
    created() {
      axios.get(`https://api.covid19api.com/total/country/jp`)
      .then(response => {
        // JSON responses are automatically parsed.
        this.japan = response.data
      })
      .catch(e => {
        this.errors.push(e)
      }),
      axios.get(`https://api.covid19api.com/total/country/uk`)
      .then(response => {
        // JSON responses are automatically parsed.
        this.uk = response.data
      })
      .catch(e => {
        this.errors.push(e)
      }),
      axios.get(`https://api.covid19api.com/total/country/nz`)
      .then(response => {
        // JSON responses are automatically parsed.
        this.newzealand = response.data
      })
      .catch(e => {
        this.errors.push(e)
      }),
      axios.get(`https://api.covid19api.com/total/country/au`)
      .then(response => {
        // JSON responses are automatically parsed.
        this.austraila = response.data
      })
      .catch(e => {
        this.errors.push(e)
      }),
      axios.get(`https://api.covid19api.com/total/country/us`)
      .then(response => {
        // JSON responses are automatically parsed.
        this.usa = response.data
      })
      .catch(e => {
        this.errors.push(e)
      })
    }
}
</script>
