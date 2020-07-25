<template>
    <v-container>
        <carousel v-bind:items="images"></carousel>    
    </v-container>
</template>

<script>
import Carousel from '../components/Carousel.vue'
export default {
  name: 'Sentiment',
  components: {
    Carousel
  },
    data () {
        return {
            images:[
                {
                    src: 'Figure_6.jpg',
                },
                {
                    src: 'Figure_7.jpg',
                },
                {
                    src: 'Figure_8.jpg',
                },
                
            ],
        }
    },
}
</script>