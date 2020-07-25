<template>
  <v-container>
    <carousel v-bind:items="items"></carousel>
  </v-container>
</template>

<script>
// @ is an alias to /src

import Carousel from '../components/Carousel.vue'
export default {
  name: 'Home',
  components: {
    Carousel
  },
  data () {
    return {
      items: [
        {
          src: 'Figure_1.jpg',
        },
        {
          src: 'Figure_2.jpg',
        },
        {
          src: 'Figure_3.jpg',
        },
        {
          src: 'Figure_4.jpg',
        },
          {
          src: 'Figure_5.jpg',
        },
      ],
    }
  },
}
</script>
